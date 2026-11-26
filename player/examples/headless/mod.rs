//! The window's view with no window, as the window has it each frame:
//! built ([`Gui::view`]), laid out, its redraw handed over and drawn, by
//! iced's headless renderer: wgpu where there is a GPU, else its CPU
//! renderer (`ICED_TEST_BACKEND=vello-cpu` asks for that one).

// Each example takes what it needs of it.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use iced::theme::Base as _;
use iced_test::core::renderer::{self, Headless as _};
use iced_test::core::{mouse, shell, window, Event, Size};
use iced_test::runtime::user_interface::{Cache, UserInterface};
use jodeldiplom::gui::window::{Gui, Message};

/// The theme the window draws in.
const THEME: iced::Theme = iced::Theme::Dark;

/// iced's renderer with no window, and what its last frame kept: the
/// widgets' state and the window's size.
pub struct Headless {
    renderer: iced::Renderer,
    cache: Cache,
    size: Size,
}

/// How long one frame of the view took, part by part.
#[derive(Clone, Copy, Debug)]
pub struct Took {
    /// [`Gui::view`].
    pub view: Duration,
    /// Its widgets' tree brought up to date and laid out.
    pub layout: Duration,
    /// The redraw handed over and the widgets drawn (iced's own lists of
    /// what to draw: the GPU is not waited for).
    pub draw: Duration,
}

impl Headless {
    /// A renderer with the fonts the player's window has.
    pub fn new() -> Headless {
        let settings = jodeldiplom::gui::iced_settings();
        for font in settings.fonts {
            iced::advanced::graphics::text::font_system()
                .write()
                .expect("the font system")
                .load_font(font);
        }
        let backend = std::env::var("ICED_TEST_BACKEND").ok();
        let renderer = pollster::block_on(iced::Renderer::new(
            renderer::Settings {
                font: settings.font,
                text_size: settings.text_size,
                line_height: settings.line_height,
                metrics_hinting: settings.metrics_hinting,
            },
            backend.as_deref(),
        ))
        .expect("a headless renderer");
        Headless {
            renderer,
            cache: Cache::default(),
            size: Size::ZERO,
        }
    }

    /// Which renderer draws: "wgpu" or "vello-cpu".
    pub fn name(&self) -> String {
        self.renderer.name()
    }

    /// One frame of `gui`'s view in a window of `size` (logical pixels),
    /// as the window draws it.
    pub fn frame(&mut self, gui: &Gui, size: Size) -> Took {
        self.size = size;
        let started = Instant::now();
        let view = gui.view();
        let viewed = Instant::now();
        let cache = std::mem::take(&mut self.cache);
        let mut ui = UserInterface::build(view, size, cache, &mut self.renderer);
        let laid = Instant::now();
        let mut messages = shell::Bus::<Message>::new();
        let redraw = Event::Window(window::Event::RedrawRequested(Instant::now()));
        let cursor = mouse::Cursor::Unavailable;
        let _ = ui.update(
            &window::Headless,
            &shell::Waker::noop(),
            &[redraw],
            cursor,
            &mut self.renderer,
            &mut messages,
        );
        let style = renderer::Style {
            text_color: THEME.base().text_color,
        };
        ui.draw(&mut self.renderer, &THEME, &style, cursor);
        let drawn = Instant::now();
        self.cache = ui.into_cache();
        Took {
            view: viewed - started,
            layout: laid - viewed,
            draw: drawn - laid,
        }
    }

    /// The last frame drawn, rendered at `scale`.
    pub fn screenshot(&mut self, scale: f32) -> image::RgbaImage {
        let (w, h) = (
            (self.size.width * scale).round() as u32,
            (self.size.height * scale).round() as u32,
        );
        let background = THEME.base().background_color;
        let rgba = self.renderer.screenshot(Size::new(w, h), scale, background);
        image::RgbaImage::from_raw(w, h, rgba).expect("the screenshot's size")
    }
}
