//! Items that take a click inside a card: a row ([`item`]: a setting, a
//! device, a folder) and a segment of a segmented control ([`segment`],
//! put together by [`segmented`]). Under the pointer an item is raised
//! (eased in over [`look::motion::HOVER`]); a selected row is the selected
//! surface with the accent's bar at its left, a chosen segment is lit in
//! the accent. What is in it draws in its ink unless it has colours of
//! its own.
//!
//! A press on what is in it that takes the press (a segment in a row) is
//! that one's alone (a chosen segment's too, though it sends nothing);
//! otherwise the item sends its message when let go of over it, as a
//! button does.

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{tree, Operation, Tree, Widget};
use iced::advanced::{mouse, overlay, Renderer as _, Shell};
use iced::animation::Easing;
use iced::widget::{container, text, Id, Row};
use iced::{
    window, Alignment, Animation, Background, Border, Color, Element, Event, Length, Padding,
    Rectangle, Renderer, Size, Theme, Vector, Widget as _,
};

use crate::gui::look::{self, height, radius, size, space};
use crate::gui::style::mix;

/// What an item is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A row of a card.
    Row,
    /// A segment of a segmented control.
    Segment,
}

/// The width of the selected row's bar, in points.
const BAR: f32 = 3.0;

/// A row of a card holding `content`, the whole width.
pub fn item<'a, Message>(content: Element<'a, Message>) -> Item<'a, Message> {
    Item {
        kind: Kind::Row,
        content,
        on_press: None,
        selected: false,
        id: None,
        width: Length::Fill,
        padding: Padding::from([space::S, space::M]),
    }
}

/// A segment saying `label`.
pub fn segment<'a, Message: 'a>(label: impl Into<String>) -> Item<'a, Message> {
    let label = text(label.into())
        .size(size::LABEL)
        .font(look::font::label())
        .wrapping(text::Wrapping::None);
    Item {
        kind: Kind::Segment,
        content: container(label).center_y(height::CHIP).boxed(),
        on_press: None,
        selected: false,
        id: None,
        width: Length::Shrink,
        padding: Padding::from([0.0, space::M]),
    }
}

/// `segments` side by side in a pill-shaped track: a segmented control.
pub fn segmented<'a, Message: Clone + 'a>(
    segments: impl IntoIterator<Item = Item<'a, Message>>,
) -> Element<'a, Message> {
    let row = segments
        .into_iter()
        .fold(Row::new().spacing(2.0), |row, s| row.push(s));
    container(row.align_y(Alignment::Center))
        .padding(2.0)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: radius::PILL.into(),
            },
            ..container::Style::default()
        })
        .boxed()
}

/// An item, as it is put together; it becomes a widget as an
/// [`Element`].
pub struct Item<'a, Message> {
    kind: Kind,
    content: Element<'a, Message>,
    on_press: Option<Message>,
    selected: bool,
    id: Option<Id>,
    width: Length,
    padding: Padding,
}

impl<Message> Item<'_, Message> {
    /// Sends `message` when clicked; without one it takes no click.
    pub fn on_press(mut self, message: Message) -> Self {
        self.on_press = Some(message);
        self
    }

    /// Sends `message` when clicked, where there is one.
    pub fn on_press_maybe(mut self, message: Option<Message>) -> Self {
        self.on_press = message;
        self
    }

    /// Selected (a row) or chosen (a segment).
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Found by `id` (the tests find an item so).
    pub fn id(mut self, id: impl Into<Id>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Its padding round what is in it.
    pub fn padding(mut self, padding: impl Into<Padding>) -> Self {
        self.padding = padding.into();
        self
    }
}

/// The colours an item is drawn in.
struct Paint {
    fill: Color,
    ink: Color,
    edge: Color,
}

/// What `kind` looks like `hover` (0 to 1) under the pointer, selected
/// or not, pressed or not.
fn paint(kind: Kind, hover: f32, selected: bool, pressed: bool) -> Paint {
    let clear = Color::TRANSPARENT;
    let (rest, over) = match (kind, selected) {
        (Kind::Row, false) => (
            (look::alpha(look::RAISED, 0.0), look::TEXT),
            (look::RAISED, look::TEXT),
        ),
        (Kind::Row, true) => (
            (look::SELECTED, look::TEXT),
            (mix(look::SELECTED, look::TEXT, 0.04), look::TEXT),
        ),
        (Kind::Segment, false) => (
            (look::alpha(look::RAISED, 0.0), look::TEXT_DIM),
            (look::RAISED, look::TEXT),
        ),
        (Kind::Segment, true) => (
            (look::tint(look::ACCENT, look::BG, 0.16), look::ACCENT),
            (look::tint(look::ACCENT, look::BG, 0.16), look::ACCENT),
        ),
    };
    let t = hover.clamp(0.0, 1.0);
    let mut fill = mix(rest.0, over.0, t);
    if pressed {
        fill = mix(look::RAISED, look::BG, 0.5);
    }
    let edge = match (kind, selected) {
        (Kind::Segment, true) => look::tint(look::ACCENT, look::BG, 0.4),
        _ => clear,
    };
    Paint {
        fill,
        ink: mix(rest.1, over.1, t),
        edge,
    }
}

/// What an item keeps between frames.
struct State {
    hover: Animation<bool>,
    pressed: bool,
    /// The redraw being drawn.
    now: Instant,
}

impl Default for State {
    fn default() -> Self {
        State {
            hover: Animation::new(false)
                .duration(look::motion::HOVER)
                .easing(Easing::EaseOut),
            pressed: false,
            now: Instant::now(),
        }
    }
}

impl<'a, Message: Clone + 'a> From<Item<'a, Message>> for Element<'a, Message> {
    fn from(item: Item<'a, Message>) -> Self {
        item.boxed()
    }
}

impl<Message> iced::advanced::widget::Meta for Item<'_, Message> {}

impl<'a, Message: Clone + 'a> Widget<Message, Theme, Renderer> for Item<'a, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_mut(&mut self.content));
    }

    fn size(&self) -> Size<Length> {
        Size::new(self.width, Length::Shrink)
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        layout::padded(
            tree,
            limits,
            self.width,
            Length::Shrink,
            self.padding,
            |content, limits| {
                self.content.layout(content, renderer, limits);
                content.size
            },
        );
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout,
        viewport: &Rectangle,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.container(self.id.as_ref(), layout.bounds(), viewport);
        operation.traverse(&mut |operation| {
            let (layout, tree) = layout
                .iter_mut(&mut tree.children)
                .next()
                .expect("the content");
            self.content
                .operate(tree, layout, viewport, renderer, operation);
        });
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        {
            let (content_layout, content_tree) = layout
                .iter_mut(&mut tree.children)
                .next()
                .expect("the content");
            self.content.update(
                content_tree,
                event,
                content_layout,
                cursor,
                renderer,
                shell,
                viewport,
            );
        }
        let bounds = layout.bounds();
        let state = tree.state.downcast_mut::<State>();
        if let Event::Window(window::Event::RedrawRequested(at)) = event {
            state.now = *at;
            if state.hover.is_animating(*at) {
                shell.request_redraw();
            }
        }
        // What is in it took the press: the pointer is that one's.
        let captured = shell.is_event_captured();
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                if !captured && self.on_press.is_some() && cursor.is_over(bounds) =>
            {
                state.pressed = true;
                shell.capture_event();
                shell.request_redraw();
            }
            // The chosen segment: nothing to do, and not the row's either.
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                if !captured
                    && self.kind == Kind::Segment
                    && self.selected
                    && cursor.is_over(bounds) =>
            {
                shell.capture_event();
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) if state.pressed => {
                state.pressed = false;
                shell.request_redraw();
                if let Some(message) = self.on_press.clone().filter(|_| cursor.is_over(bounds)) {
                    shell.publish(message);
                }
            }
            _ => {}
        }
        let over = self.on_press.is_some() && cursor.is_over(bounds);
        if state.hover.value() != over {
            state.hover.go_mut(over, Instant::now());
            shell.request_redraw();
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = tree.state.downcast_ref::<State>();
        let hover = state.hover.interpolate(0.0, 1.0, state.now);
        let paint = paint(self.kind, hover, self.selected, state.pressed);
        let corner = match self.kind {
            Kind::Row => radius::CONTROL,
            Kind::Segment => radius::PILL,
        };
        if paint.fill.a > 0.0 || paint.edge.a > 0.0 {
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Border {
                        color: paint.edge,
                        width: if paint.edge.a > 0.0 {
                            look::line::EDGE
                        } else {
                            0.0
                        },
                        radius: corner.into(),
                    },
                    ..Quad::default()
                },
                Background::Color(paint.fill),
            );
        }
        if self.kind == Kind::Row && self.selected {
            let inset = space::S;
            renderer.fill_quad(
                Quad {
                    bounds: Rectangle {
                        x: bounds.x,
                        y: bounds.y + inset,
                        width: BAR,
                        height: (bounds.height - 2.0 * inset).max(0.0),
                    },
                    border: Border {
                        radius: radius::PILL.into(),
                        ..Border::default()
                    },
                    ..Quad::default()
                },
                Background::Color(look::ACCENT),
            );
        }
        let (content_layout, content_tree) =
            layout.iter(&tree.children).next().expect("the content");
        self.content.draw(
            content_tree,
            renderer,
            theme,
            &renderer::Style {
                text_color: paint.ink,
            },
            content_layout,
            cursor,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let (content_layout, content_tree) =
            layout.iter(&tree.children).next().expect("the content");
        let inner = self.content.mouse_interaction(
            content_tree,
            content_layout,
            cursor,
            viewport,
            renderer,
        );
        if inner != mouse::Interaction::default() && inner != mouse::Interaction::None {
            inner
        } else if self.on_press.is_some() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
        window: Size,
    ) -> Vec<overlay::Element<'b, Message, Theme, Renderer>> {
        let (layout, tree) = layout
            .iter_mut(&mut tree.children)
            .next()
            .expect("the content");
        self.content
            .overlay(tree, layout, renderer, viewport, translation, window)
    }
}
