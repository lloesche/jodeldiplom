//! A list's row ([`list_row`]): what it shows laid out in it (an icon,
//! a title, its details at the right), [`look::height::ROW`] high, on
//! nothing at rest, `raised` under the pointer and `selected` with the
//! accent's bar at its left while it is the row selected, both eased in
//! over [`look::motion::HOVER`].
//!
//! It says what the pointer does over it: a press (and the grab a press
//! starts, with the row's bounds), a right press, and a move over it.
//! The list draws only the rows in view, so a long list costs no more
//! than a short one.

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{tree, Operation, Tree, Widget};
use iced::advanced::{mouse, overlay, Renderer as _, Shell};
use iced::animation::Easing;
use iced::{
    window, Animation, Background, Border, Color, Element, Event, Length, Padding, Point,
    Rectangle, Renderer, Size, Theme, Vector,
};

use crate::gui::look::{self, height, radius, space};
use crate::gui::style::mix;

/// The accent's bar at the selected row's left, in points.
const BAR: f32 = 3.0;

/// A row showing `content`.
pub fn list_row<'a, Message>(content: impl Into<Element<'a, Message>>) -> ListRow<'a, Message> {
    ListRow {
        content: content.into(),
        selected: false,
        on_press: None,
        on_grab: None,
        on_right_press: None,
        on_move: None,
        ground: look::SURFACE,
    }
}

/// A row, as it is put together; it becomes a widget as an [`Element`].
pub struct ListRow<'a, Message> {
    content: Element<'a, Message>,
    selected: bool,
    on_press: Option<Message>,
    on_grab: Option<Box<dyn Fn(Rectangle, Point) -> Message + 'a>>,
    on_right_press: Option<Message>,
    on_move: Option<Message>,
    ground: Color,
}

impl<'a, Message> ListRow<'a, Message> {
    /// The row selected: `selected` with the accent's bar.
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Sends `message` when pressed.
    pub fn on_press(mut self, message: Message) -> Self {
        self.on_press = Some(message);
        self
    }

    /// Sends `grab(bounds, at)` when pressed at `at` (after
    /// [`ListRow::on_press`]'s), `bounds` the row's in the window: a press
    /// that holds the pointer.
    pub fn on_grab(mut self, grab: impl Fn(Rectangle, Point) -> Message + 'a) -> Self {
        self.on_grab = Some(Box::new(grab));
        self
    }

    /// Sends `message` when the secondary button is pressed on it.
    pub fn on_right_press(mut self, message: Message) -> Self {
        self.on_right_press = Some(message);
        self
    }

    /// Sends `message` when the pointer moves over it (once until it
    /// leaves).
    pub fn on_move(mut self, message: Message) -> Self {
        self.on_move = Some(message);
        self
    }

    /// On `ground` (a panel's `surface` otherwise): under the pointer it
    /// lightens from there.
    pub fn ground(mut self, ground: Color) -> Self {
        self.ground = ground;
        self
    }
}

/// What a row keeps between frames.
struct State {
    hover: Animation<bool>,
    selected: Animation<bool>,
    /// The move over it was told (until the pointer leaves).
    moved: bool,
    /// The redraw being drawn.
    now: Instant,
}

impl<Message> iced::advanced::widget::Meta for ListRow<'_, Message> {}

impl<'a, Message: Clone + 'a> Widget<Message, Theme, Renderer> for ListRow<'a, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    /// A row comes in as it is: selected at once if it is the row
    /// selected (scrolled into view, it does not fade in).
    fn state(&self) -> tree::State {
        let ease = |on| {
            Animation::new(on)
                .duration(look::motion::HOVER)
                .easing(Easing::EaseOut)
        };
        tree::State::new(State {
            hover: ease(false),
            selected: ease(self.selected),
            moved: false,
            now: Instant::now(),
        })
    }

    /// Selected or no longer: it eases there (the redraws that follow
    /// move it).
    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_mut(&mut self.content));
        let state = tree.state.downcast_mut::<State>();
        if state.selected.value() != self.selected {
            state.selected.go_mut(self.selected, Instant::now());
        }
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fixed(height::ROW))
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        let padding = Padding::from([0.0, space::M]);
        layout::padded(
            tree,
            limits,
            Length::Fill,
            Length::Fixed(height::ROW),
            padding,
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
        operation.container(None, layout.bounds(), viewport);
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
        if shell.is_event_captured() {
            return;
        }
        let bounds = layout.bounds();
        let state = tree.state.downcast_mut::<State>();
        let over = cursor.position_over(bounds);
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(at) = over {
                    if let Some(message) = &self.on_press {
                        shell.publish(message.clone());
                    }
                    if let Some(grab) = &self.on_grab {
                        shell.publish(grab(bounds, at));
                    }
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                if let (Some(message), Some(_)) = (&self.on_right_press, over) {
                    shell.publish(message.clone());
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if over.is_none() {
                    state.moved = false;
                } else if let (Some(message), false) = (&self.on_move, state.moved) {
                    state.moved = true;
                    shell.publish(message.clone());
                }
            }
            Event::Window(window::Event::RedrawRequested(at)) => {
                state.now = *at;
                if state.hover.is_animating(*at) || state.selected.is_animating(*at) {
                    shell.request_redraw();
                }
            }
            _ => {}
        }
        let now = Instant::now();
        if state.hover.value() != over.is_some() {
            state.hover.go_mut(over.is_some(), now);
            shell.request_redraw();
        }
        if state.selected.value() != self.selected {
            state.selected.go_mut(self.selected, now);
            shell.request_redraw();
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        if bounds.intersection(viewport).is_none() {
            return;
        }
        let state = tree.state.downcast_ref::<State>();
        let hover = state.hover.interpolate(0.0, 1.0, state.now);
        let selected = state.selected.interpolate(0.0, 1.0, state.now);
        // Under the pointer half way from the ground to the selected
        // row's (`raised` on a panel).
        let over = mix(self.ground, look::SELECTED, 0.5);
        let fill = mix(
            mix(look::alpha(over, 0.0), over, hover),
            look::SELECTED,
            selected,
        );
        if fill.a > 0.0 {
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Border {
                        radius: radius::CONTROL.into(),
                        ..Border::default()
                    },
                    ..Quad::default()
                },
                Background::Color(fill),
            );
        }
        if selected > 0.0 {
            let inset = (bounds.height * 0.25).round();
            renderer.fill_quad(
                Quad {
                    bounds: Rectangle {
                        x: bounds.x,
                        y: bounds.y + inset,
                        width: BAR,
                        height: bounds.height - 2.0 * inset,
                    },
                    border: Border {
                        radius: radius::PILL.into(),
                        ..Border::default()
                    },
                    ..Quad::default()
                },
                Background::Color(look::alpha(look::ACCENT, selected)),
            );
        }
        let (content_layout, content_tree) =
            layout.iter(&tree.children).next().expect("the content");
        self.content.draw(
            content_tree,
            renderer,
            theme,
            style,
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
        self.content
            .mouse_interaction(content_tree, content_layout, cursor, viewport, renderer)
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

impl<'a, Message: Clone + 'a> From<ListRow<'a, Message>> for Element<'a, Message> {
    fn from(row: ListRow<'a, Message>) -> Self {
        iced::Widget::boxed(row)
    }
}
