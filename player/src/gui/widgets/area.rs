//! An area that takes the pointer and draws nothing ([`area`]): over a
//! picture drawn by something else (the Transition tab's lanes), it says
//! where a button went down and where the wheel turned, with its bounds,
//! so the window turns that into the core's actions (a grab for a drag,
//! [`crate::gui::pointer::Grab`]).

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{Id, Operation, Tree, Widget};
use iced::advanced::{mouse, Shell};
use iced::mouse::ScrollDelta;
use iced::{Element, Event, Length, Point, Rectangle, Renderer, Size, Theme, Widget as _};

/// What a press says: the area's bounds in the window and where the
/// pointer is.
type Said<'a, Message> = Box<dyn Fn(Rectangle, Point) -> Message + 'a>;
/// What a turn of the wheel says: the same, and the turn.
type Turned<'a, Message> = Box<dyn Fn(Rectangle, Point, ScrollDelta) -> Message + 'a>;

/// An area filling what it is given, taking nothing yet.
pub fn area<'a, Message>() -> Area<'a, Message> {
    Area {
        on_press: None,
        on_right_press: None,
        on_scroll: None,
        id: None,
    }
}

/// An area, as it is put together; it becomes a widget as an [`Element`].
pub struct Area<'a, Message> {
    on_press: Option<Said<'a, Message>>,
    on_right_press: Option<Said<'a, Message>>,
    on_scroll: Option<Turned<'a, Message>>,
    id: Option<Id>,
}

impl<'a, Message> Area<'a, Message> {
    /// Sends `said(bounds, at)` when the main button goes down at `at`.
    pub fn on_press(mut self, said: impl Fn(Rectangle, Point) -> Message + 'a) -> Self {
        self.on_press = Some(Box::new(said));
        self
    }

    /// Sends `said(bounds, at)` when the secondary button goes down at
    /// `at`.
    pub fn on_right_press(mut self, said: impl Fn(Rectangle, Point) -> Message + 'a) -> Self {
        self.on_right_press = Some(Box::new(said));
        self
    }

    /// Sends `said(bounds, at, delta)` when the wheel turns by `delta`
    /// with the pointer at `at`.
    pub fn on_scroll(
        mut self,
        said: impl Fn(Rectangle, Point, ScrollDelta) -> Message + 'a,
    ) -> Self {
        self.on_scroll = Some(Box::new(said));
        self
    }

    /// Found by `id` (the tests find it so).
    pub fn id(mut self, id: impl Into<Id>) -> Self {
        self.id = Some(id.into());
        self
    }
}

impl<Message> iced::advanced::widget::Meta for Area<'_, Message> {}

impl<Message> Widget<Message, Theme, Renderer> for Area<'_, Message> {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(&mut self, tree: &mut Tree, _renderer: &Renderer, limits: &layout::Limits) {
        tree.size = limits.resolve(Length::Fill, Length::Fill, Size::ZERO);
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
        _tree: &mut Tree,
        event: &Event,
        layout: Layout,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let Some(at) = cursor.position_over(bounds) else {
            return;
        };
        let message = match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                self.on_press.as_ref().map(|said| said(bounds, at))
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                self.on_right_press.as_ref().map(|said| said(bounds, at))
            }
            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                self.on_scroll.as_ref().map(|said| said(bounds, at, *delta))
            }
            _ => None,
        };
        if let Some(message) = message {
            shell.publish(message);
            shell.capture_event();
        }
    }

    fn draw(
        &self,
        _tree: &Tree,
        _renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        _layout: Layout,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
    }
}

impl<'a, Message: 'a> From<Area<'a, Message>> for Element<'a, Message> {
    fn from(area: Area<'a, Message>) -> Self {
        area.boxed()
    }
}
