//! The old canvas (the ratatui widgets rasterised into a pixmap) and the
//! one widget that would show it in the window. No screen shows it any
//! more: every one is iced's widgets, Karaoke mode a widget of its own
//! ([`super::stage_view`]); its cells' size is still the lanes' letters'
//! ([`Embed::metrics`]).
//!
//! On the GPU the pixmap goes through iced's shader widget
//! ([`super::embed_gpu`]): only the pixels that changed are sent over.
//! With iced's CPU renderer (no GPU) the pixmap is an image.
//!
//! The widget fills the window from its top left corner (the GPU's pass
//! draws in window pixels). The pointer over it goes to the window as it
//! is ([`Pointer`]), in cells with the share of one (a slider follows it
//! to the pixel); the window keeps what is held ([`super::pointer`]).
//!
//! With the CPU renderer the whole pixmap goes over as a new image each
//! time a pixel changed: iced's images cannot be updated in part.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use iced::advanced::image::{self as image, Renderer as _};
use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Renderer as _};
use iced::advanced::widget::{self, tree, Tree, Widget};
use iced::advanced::{mouse, Shell};
use iced::widget::shader::{self, Shader};
use iced::{Element, Event, Fill, Rectangle, Size, Theme, Widget as _};

use super::canvas::{Canvas, Changes};
use super::embed_gpu::Frame;
use super::lanes::Metrics;
use super::pointer::{Pointer, PointerKind};
use super::window::Message;
use crate::view::layout::Grid;
use crate::view::mouse::MouseButton;

/// The canvas and what is waiting for the next frame on screen.
pub(super) struct Shared {
    pub(super) canvas: Canvas,
    /// The pixels changed since the GPU last took them.
    pub(super) changes: Changes,
    /// Whether iced draws on the GPU (its shader widget works); `None`
    /// until the first frame was drawn.
    pub(super) gpu: Option<bool>,
    /// The window's scale factor.
    pub(super) scale: f32,
    /// The pixmap as an image, for the CPU renderer; made again when it
    /// changed.
    image: Option<image::Handle>,
    /// When the frame before the last on the GPU had not got through by
    /// the time the last began: background GPU work gives way then.
    pub(super) behind: bool,
    /// How long the last frame took to hand over to the GPU.
    pub(super) handed: Duration,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("grid", &self.canvas.grid())
            .field("gpu", &self.gpu)
            .finish_non_exhaustive()
    }
}

/// The old canvas, shared with the frames iced draws.
#[derive(Clone, Debug)]
pub struct Embed(Arc<Mutex<Shared>>);

impl Default for Embed {
    /// A canvas for a window at scale factor 1, until it says.
    fn default() -> Embed {
        Embed(Arc::new(Mutex::new(Shared {
            canvas: Canvas::new(80, 24, super::font_px(1.0)),
            changes: Changes::default(),
            gpu: None,
            scale: 1.0,
            image: None,
            behind: false,
            handed: Duration::ZERO,
        })))
    }
}

impl Embed {
    pub(super) fn lock(&self) -> MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Draws a frame with `draw` (on the canvas, at `scale`; whether the
    /// GPU shows it): the pixels it changed wait for the next frame on
    /// screen.
    pub fn draw(&self, scale: f32, draw: impl FnOnce(&mut Canvas, bool)) {
        let mut shared = self.lock();
        let shared = &mut *shared;
        shared.scale = scale;
        shared.canvas.set_font_px(super::font_px(f64::from(scale)));
        // Until iced says which renderer it has: as if on the GPU.
        let gpu = shared.gpu.unwrap_or(true);
        draw(&mut shared.canvas, gpu);
        let changes = shared.canvas.take_changes();
        if !changes.is_empty() {
            shared.image = None;
        }
        shared.changes.merge(changes);
    }

    /// Whether iced draws on the GPU (until it has drawn: as if it did).
    pub fn gpu(&self) -> bool {
        self.lock().gpu.unwrap_or(true)
    }

    /// Whether the frame before the last was late on the GPU, and how
    /// long the last took to hand over.
    pub fn late(&self) -> (bool, Duration) {
        let shared = self.lock();
        (shared.behind, shared.handed)
    }

    /// The sizes the lanes are painted at in a window of `scale` pixels a
    /// point: the canvas's cells' at that scale ([`Canvas::metrics`]).
    pub fn metrics(&self, scale: f32) -> Metrics {
        let mut shared = self.lock();
        shared.canvas.set_font_px(super::font_px(f64::from(scale)));
        shared.canvas.metrics()
    }

    /// The canvas's grid of cells.
    pub fn grid(&self) -> Grid {
        self.lock().canvas.grid()
    }

    /// The widget showing it, filling the window.
    pub fn view(&self) -> Element<'static, Message> {
        Embedded {
            shader: iced::widget::shader(Input(self.clone()))
                .width(Fill)
                .height(Fill),
            embed: self.clone(),
        }
        .boxed()
    }

    /// The pixmap as RGBA bytes, with its width and height.
    fn rgba(&self) -> (Vec<u8>, u32, u32) {
        let shared = self.lock();
        let (pixels, w, h) = shared.canvas.pixels();
        let bytes = pixels
            .iter()
            .flat_map(|&word| {
                let [r, g, b] = super::pixels::unpack(word);
                [r, g, b, 255]
            })
            .collect();
        (bytes, w as u32, h as u32)
    }
}

/// The widget: iced's shader widget on the GPU, an image on the CPU
/// renderer; the shader widget's program takes the pointer either way.
struct Embedded {
    shader: Shader<Message, Input>,
    embed: Embed,
}

impl widget::Meta for Embedded {}

impl Embedded {
    fn inner(&self) -> &dyn Widget<Message, Theme, iced::Renderer> {
        &self.shader
    }

    fn inner_mut(&mut self) -> &mut dyn Widget<Message, Theme, iced::Renderer> {
        &mut self.shader
    }
}

impl Widget<Message, Theme, iced::Renderer> for Embedded {
    fn size(&self) -> Size<iced::Length> {
        self.inner().size()
    }

    fn tag(&self) -> tree::Tag {
        self.inner().tag()
    }

    fn state(&self) -> tree::State {
        self.inner().state()
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &iced::Renderer, limits: &layout::Limits) {
        self.inner_mut().layout(tree, renderer, limits);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout,
        cursor: mouse::Cursor,
        renderer: &iced::Renderer,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.inner_mut()
            .update(tree, event, layout, cursor, renderer, shell, viewport);
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let gpu = matches!(renderer, iced::Renderer::Primary(_));
        self.embed.lock().gpu = Some(gpu);
        if gpu {
            self.inner()
                .draw(tree, renderer, theme, style, layout, cursor, viewport);
            return;
        }
        let bounds = layout.bounds();
        let bg = super::style::BG;
        renderer.fill_quad(
            renderer::Quad {
                bounds,
                ..renderer::Quad::default()
            },
            bg,
        );
        let (handle, w, h, scale) = {
            let mut shared = self.embed.lock();
            // The whole pixmap goes each time: nothing to send over.
            shared.changes = Changes::default();
            let (_, w, h) = shared.canvas.pixels();
            (shared.image.clone(), w, h, shared.scale)
        };
        let handle = handle.unwrap_or_else(|| {
            let (bytes, w, h) = self.embed.rgba();
            let handle = image::Handle::from_rgba(w, h, bytes);
            self.embed.lock().image = Some(handle.clone());
            handle
        });
        let size = Size::new(w as f32 / scale, h as f32 / scale);
        renderer.draw_image(
            image::Image::new(handle).filter_method(image::FilterMethod::Nearest),
            Rectangle::new(bounds.position(), size),
            bounds,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &iced::Renderer,
    ) -> mouse::Interaction {
        self.inner()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }
}

/// The shader widget's program: the pointer as it is, in cells, and the
/// frame for the GPU.
struct Input(Embed);

impl Input {
    /// `at` (in the widget) in cells.
    fn cells(&self, at: iced::Point) -> (f64, f64) {
        let shared = self.0.lock();
        let (cw, ch) = shared.canvas.cell();
        let s = f64::from(shared.scale);
        (
            f64::from(at.x) * s / cw.max(1) as f64,
            f64::from(at.y) * s / ch.max(1) as f64,
        )
    }
}

impl shader::Program<Message> for Input {
    type State = ();
    type Primitive = Frame;

    fn update(
        &self,
        _state: &mut (),
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        let Event::Mouse(event) = event else {
            return None;
        };
        // Over the canvas, not under something iced draws over it.
        let over = cursor.is_over(bounds);
        let kind = match event {
            mouse::Event::CursorMoved { .. }
            | mouse::Event::CursorEntered
            | mouse::Event::CursorLeft => PointerKind::Moved,
            mouse::Event::ButtonPressed(b) if over => PointerKind::Down(match b {
                mouse::Button::Left => MouseButton::Left,
                mouse::Button::Right => MouseButton::Right,
                mouse::Button::Middle => MouseButton::Middle,
                _ => return None,
            }),
            mouse::Event::WheelScrolled { delta } if over => PointerKind::Wheel(*delta),
            // Releases go to the window from wherever they happen.
            _ => return None,
        };
        let at = cursor
            .observe()
            .position()
            .map_or(iced::Point::ORIGIN, |p| {
                iced::Point::new(p.x - bounds.x, p.y - bounds.y)
            });
        let (x, y) = self.cells(at);
        let pointer = Pointer { kind, x, y };
        let action = shader::Action::publish(Message::Pointer(pointer));
        Some(if over && kind != PointerKind::Moved {
            action.and_capture()
        } else {
            action
        })
    }

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> Frame {
        Frame::new(self.0.clone())
    }
}
