//! Tabs as a segmented control ([`tabs`]): the tabs' names side by side
//! on a pill-shaped track, the one shown on a pill in the accent that
//! slides to the next one shown (over [`look::motion::PANEL`]); a tab can
//! carry a count beside its name (on a small pill of its own). The names
//! take the colour of what is under them as the pill slides; the one
//! under the pointer lights up.

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{tree, Id, Operation, Tree, Widget};
use iced::advanced::{mouse, Renderer as _, Shell};
use iced::animation::Easing;
use iced::widget::text;
use iced::{
    window, Animation, Background, Border, Color, Element, Event, Length, Rectangle, Renderer,
    Size, Theme, Vector, Widget as _,
};

use crate::gui::look::{self, radius, size};
use crate::gui::style::mix;

/// The room each side of a tab's name, in points.
const PAD: f32 = 12.0;
/// Between a name and its count.
const GAP: f32 = 6.0;
/// Each side of a count, on its pill.
const COUNT_PAD: f32 = 5.0;
/// The accent's pill inside the track's edge.
const INSET: f32 = 2.0;

/// One tab: its name, a count beside it, and what pressing it sends.
pub struct Tab<Message> {
    /// Its name.
    pub label: String,
    /// A count beside the name (the queue's length).
    pub count: Option<String>,
    /// What pressing it sends.
    pub on_press: Message,
}

/// `tabs` side by side, `shown` (an index) on the accent's pill,
/// `height` points high.
pub fn tabs<'a, Message: Clone + 'a>(
    tabs: Vec<Tab<Message>>,
    shown: usize,
    height: f32,
) -> Element<'a, Message> {
    let mut parts = Vec::new();
    let mut presses = Vec::new();
    let mut counted = Vec::new();
    let name_size = if height >= 26.0 {
        size::BODY
    } else {
        size::LABEL
    };
    for tab in tabs {
        parts.push(
            text(tab.label)
                .size(name_size)
                .font(look::font::label())
                .wrapping(text::Wrapping::None)
                .boxed(),
        );
        counted.push(tab.count.is_some());
        if let Some(count) = tab.count {
            parts.push(
                text(count)
                    .size(size::CAPTION)
                    .font(look::font::numbers())
                    .wrapping(text::Wrapping::None)
                    .boxed(),
            );
        }
        presses.push(tab.on_press);
    }
    Tabs {
        parts,
        counted,
        presses,
        shown,
        height,
        id: Id::new("tabs"),
    }
    .boxed()
}

/// The widget.
struct Tabs<'a, Message> {
    /// Each tab's name, then its count where it has one.
    parts: Vec<Element<'a, Message>>,
    /// Which tabs have a count.
    counted: Vec<bool>,
    presses: Vec<Message>,
    shown: usize,
    height: f32,
    id: Id,
}

/// What the tabs keep between frames.
struct State {
    /// The accent's pill, as a tab's index (between two while it slides).
    pill: Animation<f32>,
    /// Each tab's span, from the left: where it starts and ends, and
    /// where its count's pill does.
    spans: Vec<Span>,
    /// The redraw being drawn.
    now: Instant,
}

/// Where a tab is across the track.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    from: f32,
    to: f32,
    /// Its count's pill: from and to.
    count: Option<(f32, f32)>,
}

impl<Message> Tabs<'_, Message> {
    /// The pill's span at `at` (a tab's index, or between two).
    fn pill(spans: &[Span], at: f32) -> (f32, f32) {
        let Some(last) = spans.len().checked_sub(1) else {
            return (0.0, 0.0);
        };
        let at = at.clamp(0.0, last as f32);
        let (a, b) = (at.floor() as usize, at.ceil() as usize);
        let t = at - a as f32;
        let (a, b) = (spans[a], spans[b]);
        (a.from + (b.from - a.from) * t, a.to + (b.to - a.to) * t)
    }

    /// The tab whose span is under `x` (from the track's left).
    fn at(spans: &[Span], x: f32) -> Option<usize> {
        spans.iter().position(|s| x >= s.from && x < s.to)
    }
}

impl<Message> iced::advanced::widget::Meta for Tabs<'_, Message> {}

impl<'a, Message: Clone + 'a> Widget<Message, Theme, Renderer> for Tabs<'a, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State {
            pill: Animation::new(self.shown as f32)
                .duration(look::motion::PANEL)
                .easing(Easing::EaseInOut),
            spans: Vec::new(),
            now: Instant::now(),
        })
    }

    /// Another tab shown: the pill sets off for it (the redraws that
    /// follow move it).
    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(&mut self.parts);
        let state = tree.state.downcast_mut::<State>();
        let shown = self.shown as f32;
        if state.pill.value() != shown {
            state.pill.go_mut(shown, Instant::now());
        }
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Shrink, Length::Fixed(self.height))
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        let h = self.height;
        let loose = layout::Limits::new(Size::ZERO, Size::new(f32::INFINITY, h));
        let mut spans = Vec::with_capacity(self.counted.len());
        let mut parts = self.parts.iter_mut().zip(tree.children.iter_mut());
        let mut x = 0.0;
        for &counted in &self.counted {
            let from = x;
            x += PAD;
            let Some((name, at)) = parts.next() else {
                break;
            };
            name.layout(at, renderer, &loose);
            at.translation = Vector::new(x, ((h - at.size.height) / 2.0).round());
            x += at.size.width;
            let mut count = None;
            if counted {
                if let Some((number, at)) = parts.next() {
                    number.layout(at, renderer, &loose);
                    x += GAP;
                    let width = at.size.width + 2.0 * COUNT_PAD;
                    at.translation =
                        Vector::new(x + COUNT_PAD, ((h - at.size.height) / 2.0).round());
                    count = Some((x, x + width));
                    x += width;
                }
            }
            x += PAD;
            spans.push(Span { from, to: x, count });
        }
        tree.size = limits.resolve(Length::Shrink, Length::Fixed(h), Size::new(x, h));
        tree.state.downcast_mut::<State>().spans = spans;
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout,
        viewport: &Rectangle,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.container(Some(&self.id), layout.bounds(), viewport);
        operation.traverse(&mut |operation| {
            for (part, (layout, tree)) in self
                .parts
                .iter_mut()
                .zip(layout.iter_mut(&mut tree.children))
            {
                part.operate(tree, layout, viewport, renderer, operation);
            }
        });
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
                if let Some(at) = cursor.position_over(bounds) {
                    if let Some(message) =
                        Self::at(&state.spans, at.x - bounds.x).and_then(|i| self.presses.get(i))
                    {
                        shell.publish(message.clone());
                    }
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                // The tab under the pointer lights up.
                shell.request_redraw();
            }
            Event::Window(window::Event::RedrawRequested(at)) => {
                state.now = *at;
                if state.pill.is_animating(*at) {
                    shell.request_redraw();
                }
            }
            _ => {}
        }
        let shown = self.shown as f32;
        if state.pill.value() != shown {
            state.pill.go_mut(shown, Instant::now());
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
        let pill = |from: f32, to: f32, inset: f32, edge: Color| Quad {
            bounds: Rectangle {
                x: bounds.x + from + inset,
                y: bounds.y + inset,
                width: (to - from - 2.0 * inset).max(0.0),
                height: (bounds.height - 2.0 * inset).max(0.0),
            },
            border: Border {
                color: edge,
                width: if edge.a > 0.0 { look::line::EDGE } else { 0.0 },
                radius: radius::PILL.into(),
            },
            ..Quad::default()
        };
        // The track, then the accent's pill where it is.
        renderer.fill_quad(
            pill(0.0, bounds.width, 0.0, look::EDGE),
            Background::Color(look::BG),
        );
        let (from, to) = Self::pill(&state.spans, state.pill.interpolate(state.now));
        renderer.fill_quad(
            pill(from, to, INSET, Color::TRANSPARENT),
            Background::Color(look::ACCENT),
        );
        let over = cursor
            .position_over(bounds)
            .and_then(|at| Self::at(&state.spans, at.x - bounds.x));
        let mut parts = self.parts.iter().zip(layout.iter(&tree.children));
        for (i, span) in state.spans.iter().enumerate() {
            // How much of the tab the pill covers: its names take the
            // accent's ink as it comes.
            let covered = ((to.min(span.to) - from.max(span.from))
                / (span.to - span.from).max(1.0))
            .clamp(0.0, 1.0);
            let rest = if over == Some(i) {
                look::TEXT
            } else {
                look::TEXT_DIM
            };
            let ink = mix(rest, look::ACCENT_INK, covered);
            let style = renderer::Style { text_color: ink };
            if let Some((name, (at, part))) = parts.next() {
                name.draw(part, renderer, theme, &style, at, cursor, viewport);
            }
            if let Some((a, b)) = span.count {
                renderer.fill_quad(
                    Quad {
                        bounds: Rectangle {
                            x: bounds.x + a,
                            y: bounds.y + (bounds.height * 0.2).round(),
                            width: b - a,
                            height: bounds.height - 2.0 * (bounds.height * 0.2).round(),
                        },
                        border: Border {
                            radius: radius::PILL.into(),
                            ..Border::default()
                        },
                        ..Quad::default()
                    },
                    Background::Color(look::alpha(ink, 0.16)),
                );
                if let Some((number, (at, part))) = parts.next() {
                    number.draw(part, renderer, theme, &style, at, cursor, viewport);
                }
            }
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
        if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}
