//! The terminal frontend: crossterm input, ratatui on the terminal.
//!
//! Covers are drawn with ratatui-image (Kitty, iTerm2 or Sixel graphics on
//! terminals that answer the graphics query, coloured half blocks
//! elsewhere); the spectrum with braille dots. Mouse capture is on, so
//! terminals that pass mouse events get the same clicks, drags and wheel
//! as the window.

use std::io::{self, IsTerminal, Stdout};
use std::time::{Duration, Instant};

use image::imageops::FilterType;
use image::{RgbImage, Rgba, RgbaImage};
use ratatui::backend::{CrosstermBackend, TestBackend};
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TermEvent, KeyCode as TermKey, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton as TermButton, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::{Frame, Terminal};
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};

use crate::app::App;
use crate::art::{self, Art};
use crate::cli::Args;
use crate::core::keymap::{self, Key, KeyCode, Meaning};
use crate::core::mouse::{self, MouseButton, MouseInput, MouseKind};
use crate::core::{Action, State};
use crate::spectrum::SpectrumFrame;
use crate::view::{self, Links, Pictures, Plot, Scene};

/// The longest wait for input between frames.
const FRAME: Duration = Duration::from_millis(33);

/// A cover's protocol state: the art's id, the cells it was composed for
/// (when it starts under a border line) and the protocol.
type CoverState = (u64, Option<(u16, u16)>, StatefulProtocol);

/// Covers through ratatui-image, the spectrum in braille dots.
pub struct TermPictures {
    picker: Picker,
    /// The cover's protocol state: the art, and the cells it was composed
    /// for when it starts under a border line.
    cover: Option<CoverState>,
    art: Option<Art>,
}

impl TermPictures {
    /// Pictures drawn with `picker`'s protocol.
    pub fn new(picker: Picker) -> TermPictures {
        TermPictures {
            picker,
            cover: None,
            art: None,
        }
    }

    /// Half blocks: what any terminal shows (and what the text dumps
    /// contain).
    pub fn halfblocks() -> TermPictures {
        TermPictures::new(Picker::halfblocks())
    }

    /// The cover to draw from now on.
    pub fn set_art(&mut self, art: Option<&Art>) {
        if self.art.as_ref().map(|a| a.id) != art.map(|a| a.id) {
            self.art = art.cloned();
        }
    }
}

impl Pictures for TermPictures {
    fn cover(&mut self, frame: &mut Frame, area: Rect, _background: Color, below_line: bool) {
        let Some(art) = &self.art else {
            self.cover = None;
            view::placeholder(frame, area);
            return;
        };
        let font = self.picker.font_size();
        if below_line && self.picker.protocol_type() == ProtocolType::Halfblocks {
            // Each cell has two pixels: the cover starts on the lower one
            // of the first row.
            self.cover = None;
            draw_halfblocks(frame.buffer_mut(), area, &art.image, 1);
            return;
        }
        let composed = below_line.then_some((area.width, area.height));
        let id = art.id;
        if self
            .cover
            .as_ref()
            .is_none_or(|(i, c, _)| *i != id || *c != composed)
        {
            let image = match composed {
                // Exactly the area's pixels: a transparent half cell on
                // top, the cover under it at the left.
                Some(_) => image::DynamicImage::ImageRgba8(compose_below_line(
                    &art.image,
                    area,
                    (font.width, font.height),
                )),
                None => image::DynamicImage::ImageRgb8((*art.image).clone()),
            };
            self.cover = Some((id, composed, self.picker.new_resize_protocol(image)));
        }
        let area = if below_line {
            area
        } else {
            // The image keeps its shape: centre the cells it takes.
            fit_cells(
                area,
                (art.image.width(), art.image.height()),
                (font.width, font.height),
            )
        };
        if let Some((_, _, protocol)) = &mut self.cover {
            let resize = if below_line {
                // Already the area's size: not scaled again.
                Resize::Fit(None)
            } else {
                Resize::Scale(Some(FilterType::Triangle))
            };
            frame.render_stateful_widget(StatefulImage::default().resize(resize), area, protocol);
        }
    }

    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot) {
        view::spectrum_cells(frame.buffer_mut(), plot);
    }
}

/// The cover as an image of exactly `area`'s pixels (cells of `cell`
/// pixels): transparent, with the cover fitted under the top half cell,
/// at the left.
pub fn compose_below_line(cover: &RgbImage, area: Rect, cell: (u16, u16)) -> RgbaImage {
    let (cw, ch) = (u32::from(cell.0.max(1)), u32::from(cell.1.max(1)));
    let (w, h) = (u32::from(area.width) * cw, u32::from(area.height) * ch);
    let inset = ch / 2;
    let mut out = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    if w == 0 || h <= inset {
        return out;
    }
    let (fitted, _, _) = art::fit(cover, w, h - inset);
    for (x, y, p) in fitted.enumerate_pixels() {
        if x < w && y + inset < h {
            out.put_pixel(x, y + inset, Rgba([p[0], p[1], p[2], 255]));
        }
    }
    out
}

/// The cover drawn in half blocks (two pixels a cell, `▀` with the upper
/// as the foreground, the lower as the background), fitted into `area`
/// from `skip` half rows down, at the left. Pixels outside the cover keep
/// the terminal's background.
pub fn draw_halfblocks(buf: &mut Buffer, area: Rect, cover: &RgbImage, skip: u32) {
    let (w, h) = (u32::from(area.width), u32::from(area.height) * 2);
    if w == 0 || h <= skip {
        return;
    }
    let (fitted, _, _) = art::fit(cover, w, h - skip);
    let pixel = |x: u32, y: u32| -> Option<Color> {
        let y = y.checked_sub(skip)?;
        (x < fitted.width() && y < fitted.height()).then(|| {
            let p = fitted.get_pixel(x, y);
            Color::Rgb(p[0], p[1], p[2])
        })
    };
    for row in 0..area.height {
        for col in 0..area.width {
            let (x, y) = (u32::from(col), u32::from(row) * 2);
            let (symbol, fg, bg) = match (pixel(x, y), pixel(x, y + 1)) {
                (Some(up), Some(down)) => ("▀", up, down),
                (Some(up), None) => ("▀", up, Color::Reset),
                (None, Some(down)) => ("▄", down, Color::Reset),
                (None, None) => (" ", Color::Reset, Color::Reset),
            };
            buf[(area.x + col, area.y + row)]
                .set_symbol(symbol)
                .set_fg(fg)
                .set_bg(bg);
        }
    }
}

/// The cells an image of `size` pixels takes when scaled to fit `area`
/// with cells of `cell` pixels, centred in it.
pub fn fit_cells(area: Rect, size: (u32, u32), cell: (u16, u16)) -> Rect {
    let (iw, ih) = (f64::from(size.0.max(1)), f64::from(size.1.max(1)));
    let (cw, ch) = (f64::from(cell.0.max(1)), f64::from(cell.1.max(1)));
    let scale = (f64::from(area.width) * cw / iw).min(f64::from(area.height) * ch / ih);
    let width = ((iw * scale / cw).round() as u16).clamp(1, area.width.max(1));
    let height = ((ih * scale / ch).round() as u16).clamp(1, area.height.max(1));
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width.min(area.width),
        height.min(area.height),
    )
}

/// A crossterm key as the core's [`Key`] (presses and repeats only).
pub fn key(event: KeyEvent) -> Option<Key> {
    if event.kind == KeyEventKind::Release {
        return None;
    }
    let code = match event.code {
        TermKey::Char(c) => KeyCode::Char(c),
        TermKey::Enter => KeyCode::Enter,
        TermKey::Esc => KeyCode::Esc,
        TermKey::Backspace => KeyCode::Backspace,
        TermKey::Tab => KeyCode::Tab,
        TermKey::Up => KeyCode::Up,
        TermKey::Down => KeyCode::Down,
        TermKey::Left => KeyCode::Left,
        TermKey::Right => KeyCode::Right,
        TermKey::PageUp => KeyCode::PageUp,
        TermKey::PageDown => KeyCode::PageDown,
        TermKey::Home => KeyCode::Home,
        TermKey::End => KeyCode::End,
        TermKey::Delete => KeyCode::Delete,
        _ => return None,
    };
    Some(Key {
        code,
        ctrl: event.modifiers.contains(KeyModifiers::CONTROL),
        alt: event.modifiers.contains(KeyModifiers::ALT),
        shift: !matches!(code, KeyCode::Char(_)) && event.modifiers.contains(KeyModifiers::SHIFT),
    })
}

/// A crossterm mouse event as the core's [`MouseInput`].
pub fn mouse_input(event: MouseEvent) -> Option<MouseInput> {
    let button = |b: TermButton| match b {
        TermButton::Left => MouseButton::Left,
        TermButton::Right => MouseButton::Right,
        TermButton::Middle => MouseButton::Middle,
    };
    let kind = match event.kind {
        MouseEventKind::Down(b) => MouseKind::Down(button(b)),
        MouseEventKind::Up(b) => MouseKind::Up(button(b)),
        MouseEventKind::Drag(b) => MouseKind::Drag(button(b)),
        MouseEventKind::Moved => MouseKind::Moved,
        MouseEventKind::ScrollUp => MouseKind::ScrollUp,
        MouseEventKind::ScrollDown => MouseKind::ScrollDown,
        _ => return None,
    };
    Some(MouseInput {
        kind,
        col: event.column,
        row: event.row,
    })
}

/// What a key does in `state`, as an action.
pub fn key_action(state: &State, key: Key) -> Option<Action> {
    match keymap::lookup(state, key)? {
        Meaning::Op(op) => Some(Action::Op(op)),
        Meaning::Type(c) => Some(Action::Type(c)),
    }
}

/// Draws one frame of `state` into a buffer of `cols` × `rows` and returns
/// its text (for tests and the text dumps).
pub fn render_text(
    state: &State,
    spectrum: &SpectrumFrame,
    pictures: &mut TermPictures,
    cols: u16,
    rows: u16,
) -> String {
    // The test backend cannot fail (its error type is `Infallible`).
    let Ok(mut terminal) = Terminal::new(TestBackend::new(cols, rows));
    let scene = Scene {
        state,
        spectrum,
        background: Color::Reset,
        links: Links::Underlined,
        heard: state.playing.position,
    };
    let Ok(_) = terminal.draw(|frame| view::draw(frame, &scene, pictures));
    view::buffer_text(terminal.backend().buffer())
}

/// Whether the terminal answers a graphics query and can show real images
/// (Kitty, Sixel or iTerm2). Others get the cover in half blocks. Asking a
/// terminal that does not answer would leave a reader on standard input,
/// so only known terminals are asked. Windows consoles always answer
/// (Windows Terminal with Sixel, the classic console without graphics),
/// and Windows Terminal hosting a shell as the default console sets no
/// `WT_SESSION`, so on Windows the terminal is always asked.
fn shows_images() -> bool {
    let var = |name: &str| std::env::var(name).unwrap_or_default();
    let term = var("TERM");
    let program = var("TERM_PROGRAM");
    cfg!(windows)
        || std::env::var_os("KITTY_WINDOW_ID").is_some()
        || std::env::var_os("WT_SESSION").is_some()
        || std::env::var_os("KONSOLE_VERSION").is_some()
        || ["kitty", "foot", "ghostty", "wezterm", "mlterm"]
            .iter()
            .any(|name| term.contains(name))
        || ["WezTerm", "iTerm.app", "ghostty"].contains(&program.as_str())
        || std::env::var_os("XTERM_VERSION").is_some()
        || std::env::var_os("SSH_TTY").is_some()
}

/// Asks the terminal for its name and version (XTVERSION), followed by a
/// status report request every terminal answers, so the reply always ends.
fn terminal_version() -> Option<String> {
    use std::io::{Read, Write};
    use std::sync::mpsc;

    enable_raw_mode().ok()?;
    let reply = (|| {
        let mut stdout = io::stdout();
        stdout.write_all(b"\x1b[>0q\x1b[5n").ok()?;
        stdout.flush().ok()?;
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reply = Vec::new();
            let mut byte = [0u8; 1];
            let mut stdin = io::stdin().lock();
            while !ends_with_status(&reply) && stdin.read_exact(&mut byte).is_ok() {
                reply.push(byte[0]);
            }
            let _ = sender.send(reply);
        });
        receiver.recv_timeout(Duration::from_secs(1)).ok()
    })();
    let _ = disable_raw_mode();
    version_from_reply(&reply?)
}

fn ends_with_status(reply: &[u8]) -> bool {
    reply.ends_with(b"\x1b[0n") || reply.ends_with(b"\x1b[3n")
}

fn version_from_reply(reply: &[u8]) -> Option<String> {
    let start = reply.windows(3).position(|w| w == b"\x1bP>")? + 3;
    let text = reply[start..].strip_prefix(b"|")?;
    let end = text
        .iter()
        .position(|&b| b == 0x1b || b == 0x07)
        .unwrap_or(text.len());
    Some(String::from_utf8_lossy(&text[..end]).into_owned())
}

/// The picker for this terminal: its graphics protocol, else half blocks.
fn picker() -> Picker {
    let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
    (interactive && shows_images())
        .then(|| Picker::from_query_stdio().ok())
        .flatten()
        .map(|mut picker| {
            // iTerm2 claims Kitty graphics but does not draw the
            // placeholders Kitty images are placed with; only its version
            // reply tells it apart over SSH.
            if picker.protocol_type() == ProtocolType::Kitty
                && terminal_version().is_some_and(|v| v.starts_with("iTerm2"))
            {
                picker.set_protocol_type(ProtocolType::Iterm2);
            }
            picker
        })
        .unwrap_or_else(Picker::halfblocks)
}

fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) {
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), DisableBracketedPaste);
    let _ = execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = terminal.show_cursor();
}

/// Runs the player in the terminal until it quits.
pub fn run(args: &Args) -> Result<(), String> {
    if !io::stdout().is_terminal() {
        return Err("jodeldiplom needs a terminal (or use jodeldiplom-gui)".into());
    }
    let mut options = crate::cli::app_options(args)?;
    let picker = picker();
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).map_err(|e| e.to_string())?;
    // A paste then arrives whole (`Event::Paste`) rather than as keys,
    // where the terminal can (not through the Windows console's input).
    let _ = execute!(stdout, EnableBracketedPaste);
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout)).map_err(|e| e.to_string())?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        default_hook(info);
    }));
    // The moves written down go on in the background; the files the player
    // needs, got first if they are not all here.
    crate::cli::start_mover(&mut options);
    let paths = match crate::cli::models_ready(&options) {
        Ok(paths) => paths,
        Err((manifest, places)) => match setup_screen(&mut terminal, manifest, places) {
            Ok(Some(paths)) => paths,
            Ok(None) => {
                restore(&mut terminal);
                return Ok(());
            }
            Err(error) => {
                restore(&mut terminal);
                return Err(error);
            }
        },
    };
    crate::cli::set_models(&mut options, &paths);
    let size = terminal.size().map_err(|e| e.to_string())?;
    options.size = (size.width, size.height);
    let mut app = App::new(options);
    let result = event_loop(&mut terminal, &mut app, picker, args.exit_after);
    app.shutdown();
    restore(&mut terminal);
    result
}

/// The setup screen until setup has the files (their paths), or the person
/// quits (`None`).
fn setup_screen(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    manifest: Vec<crate::setup::Need>,
    places: crate::setup::Places,
) -> Result<Option<Vec<std::path::PathBuf>>, String> {
    use crate::view::setup::{choice, draw, Buttons, Choice};
    use std::sync::Arc;
    let setup = crate::setup::Setup::start(
        manifest,
        places,
        Arc::new(crate::setup::Real),
        Arc::new(|| {}),
    );
    let mut buttons = Buttons::default();
    loop {
        let status = setup.status();
        if let Some(paths) = status.finished {
            let _ = terminal.clear();
            return Ok(Some(paths));
        }
        terminal
            .draw(|frame| buttons = draw(frame, &status, Color::Reset))
            .map_err(|e| e.to_string())?;
        if !event::poll(Duration::from_millis(100)).map_err(|e| e.to_string())? {
            continue;
        }
        let failed = status.failure.is_some();
        let chosen = match event::read().map_err(|e| e.to_string())? {
            TermEvent::Key(k) if k.kind == event::KeyEventKind::Press => {
                if k.code == event::KeyCode::Char('c')
                    && k.modifiers.contains(KeyModifiers::CONTROL)
                {
                    Some(Choice::Quit)
                } else {
                    let name = match k.code {
                        event::KeyCode::Enter => "enter".to_owned(),
                        event::KeyCode::Esc => "esc".to_owned(),
                        event::KeyCode::Char(c) => c.to_string(),
                        _ => String::new(),
                    };
                    choice(&name, failed)
                }
            }
            TermEvent::Mouse(m) if matches!(m.kind, event::MouseEventKind::Down(_)) => {
                buttons.at(m.column, m.row)
            }
            _ => None,
        };
        match chosen {
            Some(Choice::Quit) => return Ok(None),
            Some(Choice::Retry) => setup.retry(),
            None => {}
        }
    }
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    picker: Picker,
    exit_after: Option<f64>,
) -> Result<(), String> {
    let mut pictures = TermPictures::new(picker);
    let started = Instant::now();
    let mut dirty = true;
    let mut last_draw = Instant::now() - FRAME;
    let mut last_screen = app.state.screen;
    loop {
        dirty |= app.pump();
        // The size now, before drawing: a resize event can come late (or
        // not at all), and the layout must be for the frame drawn.
        let size = terminal.size().map_err(|e| e.to_string())?;
        if (size.width, size.height) != app.state.size {
            app.dispatch(Action::Resize(size.width, size.height));
            dirty = true;
        }
        if app.state.quit || exit_after.is_some_and(|s| started.elapsed().as_secs_f64() >= s) {
            return Ok(());
        }
        let animating = app.animating();
        if (dirty || animating) && last_draw.elapsed() >= FRAME {
            // Graphics protocols leave images on screen; start clean when
            // the screen changes.
            if app.state.screen != last_screen {
                let _ = terminal.clear();
                last_screen = app.state.screen;
            }
            pictures.set_art(app.art());
            let spectrum = app.spectrum();
            let heard = app.heard_for_frame();
            let scene = Scene {
                state: &app.state,
                spectrum: &spectrum,
                background: Color::Reset,
                links: Links::Underlined,
                heard,
            };
            terminal
                .draw(|frame| view::draw(frame, &scene, &mut pictures))
                .map_err(|e| e.to_string())?;
            last_draw = Instant::now();
            dirty = false;
        }
        let wait = if animating || dirty {
            FRAME.saturating_sub(last_draw.elapsed())
        } else {
            Duration::from_millis(100)
        };
        if !event::poll(wait).map_err(|e| e.to_string())? {
            continue;
        }
        while event::poll(Duration::ZERO).map_err(|e| e.to_string())? {
            let event = event::read().map_err(|e| e.to_string())?;
            // The clock first: what a key means can depend on the time
            // (space while a name is being typed).
            app.dispatch(Action::Tick(app.now_ms()));
            let action = match event {
                TermEvent::Key(k) => key(k).and_then(|k| key_action(&app.state, k)),
                TermEvent::Mouse(m) => {
                    let shift = m.modifiers.contains(KeyModifiers::SHIFT);
                    mouse_input(m)
                        .and_then(|input| mouse::action(&app.state, input))
                        .map(|a| if shift { mouse::shifted(a) } else { a })
                }
                TermEvent::Resize(cols, rows) => Some(Action::Resize(cols, rows)),
                TermEvent::Paste(text) => Some(Action::Paste(text)),
                _ => None,
            };
            if let Some(action) = action {
                app.dispatch(action);
                dirty = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_translate_and_releases_are_ignored() {
        let press = KeyEvent::new(TermKey::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(key(press), Some(Key::ctrl('c')));
        let mut release = KeyEvent::new(TermKey::Enter, KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        assert_eq!(key(release), None);
    }

    #[test]
    fn the_version_reply_names_iterm2() {
        let reply = b"\x1b_Gi=31;OK\x1b\\\x1b[?64;1;2c\x1bP>|iTerm2 3.7.0\x1b\\\x1b[0n";
        assert!(ends_with_status(reply));
        assert_eq!(version_from_reply(reply).as_deref(), Some("iTerm2 3.7.0"));
        assert_eq!(version_from_reply(b"\x1b[0n"), None);
    }
}
