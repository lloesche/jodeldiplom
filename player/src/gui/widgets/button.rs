//! Buttons: [`primary`] (the accent's fill: the one to take), [`secondary`]
//! (raised), [`ghost`] (text alone, raised under the pointer) and [`icon`]
//! (a square ghost round an icon). A label can have an icon before it
//! ([`Button::leading`]) and the key that does the same after it
//! ([`Button::key`]).
//!
//! Each has its states: under the pointer (eased in over
//! [`look::motion::HOVER`]), pressed, focused (a ring in the accent round
//! it) and disabled (no message to send).

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{tree, Operation, Tree, Widget};
use iced::advanced::{mouse, overlay, Renderer as _, Shell};
use iced::animation::Easing;
use iced::widget::{container, text, Row};
use iced::{
    window, Alignment, Animation, Background, Border, Color, Element, Event, Length, Padding,
    Rectangle, Renderer, Shadow, Size, Theme, Vector, Widget as _,
};

use super::icons::{self, Icon};
use crate::gui::look::{self, height, radius, size, space};
use crate::gui::style::mix;

/// What a button is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The accent's fill: the one to take.
    Primary,
    /// Raised.
    Secondary,
    /// Text alone, raised under the pointer.
    Ghost,
    /// A square ghost round an icon.
    Icon,
}

/// A button's state as it is drawn: how far the pointer's hover has eased
/// in (0 to 1), pressed, focused, disabled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Look {
    /// How far the pointer's hover has eased in: 0 to 1.
    pub hover: f32,
    /// Held down.
    pub pressed: bool,
    /// Ringed in the accent.
    pub focused: bool,
    /// With nothing to send.
    pub disabled: bool,
}

/// A primary button saying `label`.
pub fn primary<'a, Message>(label: impl Into<String>) -> Button<'a, Message> {
    Button::new(Kind::Primary, Some(label.into()), None)
}

/// A secondary button saying `label`.
pub fn secondary<'a, Message>(label: impl Into<String>) -> Button<'a, Message> {
    Button::new(Kind::Secondary, Some(label.into()), None)
}

/// A ghost button saying `label`.
pub fn ghost<'a, Message>(label: impl Into<String>) -> Button<'a, Message> {
    Button::new(Kind::Ghost, Some(label.into()), None)
}

/// A square button showing `icon`.
pub fn icon<'a, Message>(icon: Icon) -> Button<'a, Message> {
    Button::new(Kind::Icon, None, Some(icon))
}

/// A button, as it is put together; it becomes a widget as an
/// [`Element`].
pub struct Button<'a, Message> {
    kind: Kind,
    label: Option<String>,
    icon: Option<Icon>,
    key: Option<&'a str>,
    on_press: Option<Message>,
    focused: bool,
    shown: Option<Look>,
    width: Length,
}

impl<'a, Message> Button<'a, Message> {
    fn new(kind: Kind, label: Option<String>, icon: Option<Icon>) -> Button<'a, Message> {
        Button {
            kind,
            label,
            icon,
            key: None,
            on_press: None,
            focused: false,
            shown: None,
            width: Length::Shrink,
        }
    }

    /// `icon` before the label.
    pub fn leading(mut self, icon: Icon) -> Self {
        self.icon = Some(icon);
        self
    }

    /// The key that does the same, after the label ("Enter", "Q").
    pub fn key(mut self, key: &'a str) -> Self {
        self.key = Some(key);
        self
    }

    /// Sends `message` when pressed; without one it is disabled.
    pub fn on_press(mut self, message: Message) -> Self {
        self.on_press = Some(message);
        self
    }

    /// With the ring of the focused control round it.
    pub fn focused(mut self, focused: bool) -> Self {
        self.focused = focused;
        self
    }

    /// Drawn as `look` whatever the pointer does (the look sheet shows
    /// every state this way).
    pub fn shown(mut self, look: Look) -> Self {
        self.shown = Some(look);
        self
    }

    /// Its width (it fits its label otherwise).
    pub fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }
}

/// The key's cap after a label: small, in the button's own tone.
fn keycap<'a, Message: 'a>(key: &'a str, kind: Kind) -> Element<'a, Message> {
    let (fill, ink) = match kind {
        Kind::Primary => (
            look::alpha(look::ACCENT_INK, 0.12),
            look::alpha(look::ACCENT_INK, 0.75),
        ),
        _ => (look::alpha(look::BG, 0.7), look::TEXT_DIM),
    };
    container(
        text(key)
            .size(size::CAPTION)
            .font(look::font::label())
            .color(ink),
    )
    .padding(Padding::from([1.0, 6.0]))
    .style(move |_| container::Style {
        background: Some(Background::Color(fill)),
        border: Border {
            radius: radius::CONTROL.into(),
            ..Border::default()
        },
        ..container::Style::default()
    })
    .boxed()
}

impl<'a, Message: Clone + 'a> From<Button<'a, Message>> for Element<'a, Message> {
    fn from(button: Button<'a, Message>) -> Self {
        let Button {
            kind,
            label,
            icon,
            key,
            on_press,
            focused,
            shown,
            width,
        } = button;
        let mut parts = Row::new().spacing(space::S).align_y(Alignment::Center);
        if let Some(icon) = icon {
            parts = parts.push(icons::icon(icon).boxed());
        }
        if let Some(label) = label {
            parts = parts.push(
                text(label)
                    .size(size::BODY)
                    .font(look::font::label())
                    .wrapping(text::Wrapping::None)
                    .boxed(),
            );
        }
        if let Some(key) = key {
            parts = parts.push(keycap(key, kind));
        }
        let (width, padding) = match kind {
            Kind::Icon => (Length::Fixed(height::CONTROL), 0.0),
            _ => (width, space::L),
        };
        let content = container(parts)
            .center_y(Length::Fill)
            .align_x(Alignment::Center)
            .width(if width == Length::Shrink {
                Length::Shrink
            } else {
                Length::Fill
            });
        Pressable {
            kind,
            content: content.boxed(),
            on_press,
            focused,
            shown,
            width,
            padding,
        }
        .boxed()
    }
}

/// The colours a button is drawn in.
struct Paint {
    fill: Color,
    ink: Color,
    edge: Color,
    glow: Color,
}

/// What `kind` looks like in `look`.
fn paint(kind: Kind, look: Look) -> Paint {
    let clear = Color::TRANSPARENT;
    // At rest, and with the pointer over it.
    let (rest, over) = match kind {
        Kind::Primary => (
            Paint {
                fill: look::ACCENT,
                ink: look::ACCENT_INK,
                edge: clear,
                glow: clear,
            },
            Paint {
                fill: mix(look::ACCENT, Color::WHITE, 0.14),
                ink: look::ACCENT_INK,
                edge: clear,
                glow: look::alpha(look::ACCENT, 0.28),
            },
        ),
        Kind::Secondary => (
            Paint {
                fill: look::RAISED,
                ink: look::TEXT,
                edge: look::EDGE,
                glow: clear,
            },
            Paint {
                fill: look::SELECTED,
                ink: look::TEXT,
                edge: mix(look::EDGE, look::TEXT_FAINT, 0.4),
                glow: clear,
            },
        ),
        Kind::Ghost | Kind::Icon => (
            Paint {
                fill: look::alpha(look::RAISED, 0.0),
                ink: look::TEXT_DIM,
                edge: clear,
                glow: clear,
            },
            Paint {
                fill: look::RAISED,
                ink: look::TEXT,
                edge: clear,
                glow: clear,
            },
        ),
    };
    if look.disabled {
        return Paint {
            fill: match kind {
                Kind::Primary | Kind::Secondary => look::RAISED,
                Kind::Ghost | Kind::Icon => clear,
            },
            ink: look::TEXT_FAINT,
            edge: clear,
            glow: clear,
        };
    }
    let t = look.hover.clamp(0.0, 1.0);
    let mut paint = Paint {
        fill: mix(rest.fill, over.fill, t),
        ink: mix(rest.ink, over.ink, t),
        edge: mix(rest.edge, over.edge, t),
        glow: mix(rest.glow, over.glow, t),
    };
    if look.pressed {
        paint.fill = match kind {
            Kind::Primary => mix(look::ACCENT, Color::BLACK, 0.12),
            _ => mix(look::RAISED, look::BG, 0.5),
        };
        paint.glow = clear;
    }
    paint
}

/// The widget: its content laid out in its padding, its look eased.
struct Pressable<'a, Message> {
    kind: Kind,
    content: Element<'a, Message>,
    on_press: Option<Message>,
    focused: bool,
    shown: Option<Look>,
    width: Length,
    padding: f32,
}

/// What a button keeps between frames.
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

impl<Message> Pressable<'_, Message> {
    fn look(&self, state: &State) -> Look {
        self.shown.unwrap_or(Look {
            hover: state.hover.interpolate(0.0, 1.0, state.now),
            pressed: state.pressed,
            focused: self.focused,
            disabled: self.on_press.is_none(),
        })
    }
}

impl<Message> iced::advanced::widget::Meta for Pressable<'_, Message> {}

impl<'a, Message: Clone + 'a> Widget<Message, Theme, Renderer> for Pressable<'a, Message> {
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
        Size::new(self.width, Length::Fixed(height::CONTROL))
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        let padding = Padding::from([0.0, self.padding]);
        layout::padded(
            tree,
            limits,
            self.width,
            Length::Fixed(height::CONTROL),
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
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                if self.on_press.is_some() && cursor.is_over(bounds) =>
            {
                state.pressed = true;
                shell.capture_event();
                shell.request_redraw();
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) if state.pressed => {
                state.pressed = false;
                shell.request_redraw();
                if let Some(message) = self.on_press.clone().filter(|_| cursor.is_over(bounds)) {
                    shell.publish(message);
                }
            }
            Event::Window(window::Event::RedrawRequested(at)) => {
                state.now = *at;
                if state.hover.is_animating(*at) {
                    shell.request_redraw();
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
        let look = self.look(state);
        let paint = paint(self.kind, look);
        if look.focused {
            let gap = look::line::FOCUS;
            let ring = bounds.expand(gap + look::line::FOCUS);
            renderer.fill_quad(
                Quad {
                    bounds: ring,
                    border: Border {
                        color: look::ACCENT,
                        width: look::line::FOCUS,
                        radius: (radius::CONTROL + gap + look::line::FOCUS).into(),
                    },
                    ..Quad::default()
                },
                Background::Color(Color::TRANSPARENT),
            );
        }
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
                    radius: radius::CONTROL.into(),
                },
                shadow: Shadow {
                    color: paint.glow,
                    offset: Vector::new(0.0, 2.0),
                    blur_radius: if paint.glow.a > 0.0 { 14.0 } else { 0.0 },
                },
                ..Quad::default()
            },
            Background::Color(paint.fill),
        );
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
