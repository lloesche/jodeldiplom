//! The screens not ported to iced yet, inside its window: the old canvas
//! (the ratatui widgets rasterised into a pixmap, the pictures painted in
//! pixels) draws the player's whole screen at the window's size, and one
//! widget shows it under everything iced draws itself.
//!
//! On the GPU the pixmap goes through iced's shader widget
//! ([`super::embed_gpu`]): only the pixels that changed are sent over, the
//! lanes that follow the playhead scroll on the GPU, Karaoke mode's stage
//! is drawn over it. With iced's CPU renderer (no GPU) the pixmap is an
//! image, the lanes painted into it and no stage.
//!
//! The widget fills the window from its top left corner (the GPU's pass
//! draws in window pixels). The pointer over it goes to the window as it
//! is ([`Pointer`]), in cells with the share of one (a slider follows it
//! to the pixel), and the target drawn in pixels under it (Karaoke mode's
//! START pill, which the stage draws in pixels, not cells:
//! [`Canvas::target_at`]; the hand shows over a button there); the window
//! keeps what is held ([`super::pointer`]). Over it the pointer hides
//! where the window says (Karaoke mode, still a while:
//! [`crate::core::karaoke::pointer`]).
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

use super::backdrop::BackdropFrame;
use super::canvas::{Canvas, Changes, LayerFrame};
use super::embed_gpu::Frame;
use super::pointer::{Pointer, PointerKind};
use super::stage_gpu::StageFrame;
use super::window::Message;
use crate::core::mouse::{MouseButton, Target};

/// What a frame of the old canvas hands the GPU beside its pixels.
#[derive(Default)]
pub struct Drawn {
    /// The lane that follows the playhead, where one shows.
    pub layer: Option<LayerFrame>,
    /// Karaoke mode's stage, where it shows.
    pub stage: Option<StageFrame>,
    /// Karaoke mode's backdrop, where it shows.
    pub backdrop: Option<BackdropFrame>,
}

/// The canvas and what is waiting for the next frame on screen.
pub(super) struct Shared {
    pub(super) canvas: Canvas,
    /// The pixels changed since the GPU last took them.
    pub(super) changes: Changes,
    /// The layer and the stage as they are now; their uploads not sent yet.
    pub(super) layer: Option<LayerFrame>,
    pub(super) stage: Option<StageFrame>,
    /// The backdrop as it is now (its textures sent over as it first shows).
    pub(super) backdrop: Option<BackdropFrame>,
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
            layer: None,
            stage: None,
            backdrop: None,
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
    /// GPU shows it): the pixels it changed, and its layer and stage, wait
    /// for the next frame on screen.
    pub fn draw(&self, scale: f32, draw: impl FnOnce(&mut Canvas, bool) -> Drawn) {
        let mut shared = self.lock();
        let shared = &mut *shared;
        shared.scale = scale;
        shared.canvas.set_font_px(super::font_px(f64::from(scale)));
        // What the GPU draws itself (the lanes it scrolls, Karaoke mode's
        // ground and stage), where it shows the canvas (and until iced says
        // which renderer it has).
        let gpu = shared.gpu.unwrap_or(true);
        shared.canvas.set_gpu(gpu);
        let drawn = draw(&mut shared.canvas, gpu);
        let changes = shared.canvas.take_changes();
        if !changes.is_empty() {
            shared.image = None;
        }
        shared.changes.merge(changes);
        shared.layer = merge_layer(shared.layer.take(), drawn.layer);
        shared.stage = merge_stage(shared.stage.take(), drawn.stage);
        shared.backdrop = drawn.backdrop;
    }

    /// Whether iced draws on the GPU (until it has drawn: as if it did).
    pub fn gpu(&self) -> bool {
        self.lock().gpu.unwrap_or(true)
    }

    /// Whether a lane that follows the playhead shows now.
    pub fn has_layer(&self) -> bool {
        self.lock().canvas.has_layer()
    }

    /// Whether the canvas's pictures move by themselves
    /// ([`super::Canvas::moving`]).
    pub fn moving(&self) -> bool {
        self.lock().canvas.moving()
    }

    /// Whether the frame before the last was late on the GPU, and how
    /// long the last took to hand over.
    pub fn late(&self) -> (bool, Duration) {
        let shared = self.lock();
        (shared.behind, shared.handed)
    }

    /// The text of the canvas's cells as last drawn (the pictures' cells
    /// blank).
    pub fn text(&self) -> String {
        self.lock().canvas.text()
    }

    /// A cell's size in logical pixels.
    pub fn cell(&self) -> (f32, f32) {
        let shared = self.lock();
        let (cw, ch) = shared.canvas.cell();
        (cw as f32 / shared.scale, ch as f32 / shared.scale)
    }

    /// The widget showing it, filling the window; the pointer hidden over
    /// it if `hide_pointer`.
    pub fn view(&self, hide_pointer: bool) -> Element<'static, Message> {
        Embedded {
            shader: iced::widget::shader(Input(self.clone()))
                .width(Fill)
                .height(Fill),
            embed: self.clone(),
            hide_pointer,
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

/// `new` layer's frame after `old`'s, not yet on the GPU: `old`'s tiles
/// still to be sent over first, where the tiles are the same size.
fn merge_layer(old: Option<LayerFrame>, new: Option<LayerFrame>) -> Option<LayerFrame> {
    match (old, new) {
        (Some(old), Some(mut new))
            if (old.tile, old.rect.3, old.slots) == (new.tile, new.rect.3, new.slots) =>
        {
            let mut uploads = old.uploads;
            uploads.append(&mut new.uploads);
            new.uploads = uploads;
            Some(new)
        }
        (_, new) => new,
    }
}

/// `new` stage after `old`'s, not yet on the GPU: the text `old` put into
/// the atlas is still to be sent over first (the stage counts on it), even
/// where no stage shows now.
fn merge_stage(old: Option<StageFrame>, new: Option<StageFrame>) -> Option<StageFrame> {
    let pending = old.map(|o| o.uploads).unwrap_or_default();
    if pending.is_empty() {
        return new;
    }
    let mut new = new.unwrap_or_default();
    let mut uploads = pending;
    uploads.append(&mut new.uploads);
    new.uploads = uploads;
    Some(new)
}

/// The widget: iced's shader widget on the GPU, an image on the CPU
/// renderer; the shader widget's program takes the pointer either way.
struct Embedded {
    shader: Shader<Message, Input>,
    embed: Embed,
    /// The pointer hidden over it (over a button too).
    hide_pointer: bool,
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
        // Only over it: iced shows the pointer again wherever else it is,
        // and as soon as it is not to be hidden.
        if self.hide_pointer && cursor.is_over(layout.bounds()) {
            return mouse::Interaction::Hidden;
        }
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

    /// The target drawn in pixels at `at` (in the widget) in the frame
    /// shown.
    fn target(&self, at: iced::Point) -> Option<Target> {
        let shared = self.0.lock();
        let s = shared.scale;
        shared.canvas.target_at(at.x * s, at.y * s)
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
        let pointer = Pointer {
            kind,
            x,
            y,
            over,
            on: self.target(at).filter(|_| over),
        };
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

    /// The hand over a button drawn in pixels (START's pill).
    fn mouse_interaction(
        &self,
        _state: &(),
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        match cursor.position_in(bounds).and_then(|at| self.target(at)) {
            Some(Target::Button(_)) => mouse::Interaction::Pointer,
            _ => mouse::Interaction::default(),
        }
    }
}
