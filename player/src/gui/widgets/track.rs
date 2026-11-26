//! A track: the seek bar's or a slider's line ([`track`]), thin and
//! pill-shaped, filled from the left in the accent up to its share, with a
//! thumb there under the pointer or while it is held (easing in over
//! [`look::motion::HOVER`]). A press on it hands the window a grab with
//! the track's bounds ([`crate::gui::pointer::Grab`]): the pointer then
//! moves it from anywhere in the window, and the release lands it.

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{tree, Id, Operation, Tree, Widget};
use iced::advanced::{mouse, Renderer as _, Shell};
use iced::animation::Easing;
use iced::{
    window, Animation, Background, Border, Color, Element, Event, Length, Point, Rectangle,
    Renderer, Size, Theme, Widget as _,
};

use crate::gui::look::{self, height, radius};

/// The height that takes the pointer, in points: more than the line, so
/// it is easy to hit.
pub const REACH: f32 = 16.0;

/// The thumb's diameter, in points.
const THUMB: f32 = 12.0;

/// A track at `share` (0 to 1) of its width.
pub fn track<'a, Message>(share: f64) -> Track<'a, Message> {
    Track {
        share: share.clamp(0.0, 1.0) as f32,
        held: false,
        on_press: None,
        width: Length::Fill,
        id: None,
    }
}

/// A track, as it is put together; it becomes a widget as an [`Element`].
pub struct Track<'a, Message> {
    share: f32,
    held: bool,
    on_press: Option<Box<dyn Fn(Rectangle, Point) -> Message + 'a>>,
    width: Length,
    id: Option<Id>,
}

impl<'a, Message> Track<'a, Message> {
    /// Sends `grab(bounds, at)` when pressed at `at`, `bounds` the track's
    /// in the window (without it the track only shows its share, its
    /// thumb hidden).
    pub fn on_press(mut self, grab: impl Fn(Rectangle, Point) -> Message + 'a) -> Self {
        self.on_press = Some(Box::new(grab));
        self
    }

    /// Held: its thumb shows wherever the pointer is.
    pub fn held(mut self, held: bool) -> Self {
        self.held = held;
        self
    }

    /// Its width (all there is otherwise).
    pub fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }

    /// Found by `id` (the tests find it so).
    pub fn id(mut self, id: impl Into<Id>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// What a track keeps between frames.
struct State {
    /// The thumb easing in and out.
    thumb: Animation<bool>,
    /// The redraw being drawn.
    now: Instant,
}

impl Default for State {
    fn default() -> Self {
        State {
            thumb: Animation::new(false)
                .duration(look::motion::HOVER)
                .easing(Easing::EaseOut),
            now: Instant::now(),
        }
    }
}

impl<Message> iced::advanced::widget::Meta for Track<'_, Message> {}

impl<Message> Widget<Message, Theme, Renderer> for Track<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(self.width, Length::Fixed(REACH))
    }

    fn layout(&mut self, tree: &mut Tree, _renderer: &Renderer, limits: &layout::Limits) {
        tree.size = limits.resolve(self.width, Length::Fixed(REACH), Size::new(0.0, REACH));
    }

    fn operate(
        &mut self,
        _tree: &mut Tree,
        layout: Layout,
        viewport: &Rectangle,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.container(self.id.as_ref(), layout.bounds(), viewport);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = tree.state.downcast_mut::<State>();
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let (Some(grab), Some(at)) = (&self.on_press, cursor.position_over(bounds)) {
                    shell.publish(grab(bounds, at));
                    shell.capture_event();
                }
            }
            Event::Window(window::Event::RedrawRequested(at)) => {
                state.now = *at;
                if state.thumb.is_animating(*at) {
                    shell.request_redraw();
                }
            }
            _ => {}
        }
        let shown = self.on_press.is_some() && (self.held || cursor.is_over(bounds));
        if state.thumb.value() != shown {
            state.thumb.go_mut(shown, Instant::now());
            shell.request_redraw();
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        if bounds.intersection(viewport).is_none() {
            return;
        }
        let state = tree.state.downcast_ref::<State>();
        let thumb = state.thumb.interpolate(0.0, 1.0, state.now);
        let line = height::TRACK;
        let y = bounds.center_y() - line / 2.0;
        let pill = |x: f32, width: f32, color: Color, renderer: &mut Renderer| {
            renderer.fill_quad(
                Quad {
                    bounds: Rectangle::new(Point::new(x, y), Size::new(width, line)),
                    border: Border {
                        radius: radius::PILL.into(),
                        ..Border::default()
                    },
                    ..Quad::default()
                },
                Background::Color(color),
            );
        };
        pill(bounds.x, bounds.width, look::EDGE, renderer);
        let head = bounds.x + bounds.width * self.share;
        if self.share > 0.0 {
            pill(bounds.x, head - bounds.x, look::ACCENT, renderer);
        }
        if thumb > 0.0 {
            let d = THUMB * (0.5 + 0.5 * thumb);
            renderer.fill_quad(
                Quad {
                    bounds: Rectangle::new(
                        Point::new(head - d / 2.0, bounds.center_y() - d / 2.0),
                        Size::new(d, d),
                    ),
                    border: Border {
                        radius: radius::PILL.into(),
                        width: 2.0,
                        color: look::alpha(look::BG, 0.6 * thumb),
                    },
                    ..Quad::default()
                },
                Background::Color(look::alpha(look::ACCENT, thumb)),
            );
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        if self.on_press.is_some() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: 'a> From<Track<'a, Message>> for Element<'a, Message> {
    fn from(track: Track<'a, Message>) -> Self {
        track.boxed()
    }
}
