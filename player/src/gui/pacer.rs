//! The ticks at the display's refresh: a widget around everything the
//! window shows ([`Pacer::view`]) sends one ([`Message::Frame`]) from
//! inside each redraw, before that redraw is drawn, and asks for the next.
//! So the frame shown is the one ticked for it, once a refresh; a tick
//! sent on from the redraw (as iced's frame subscription does) lands after
//! it, and on a 60 Hz display missed every other refresh.
//!
//! The window says whether the ticks come with every redraw
//! ([`Pacer::tick_every_redraw`], from its subscription); otherwise they
//! come from its timers ([`super::paint::Pace`]).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{self, tree, Operation, Tree, Widget};
use iced::advanced::{mouse, overlay, Shell};
use iced::{Element, Event, Length, Rectangle, Renderer, Size, Theme, Vector, Widget as _};

use super::window::Message;

/// The shortest time between two ticks at the refresh: 250 a second, past
/// any display's refresh worth following.
const FASTEST: Duration = Duration::from_millis(4);

/// Whether a tick comes with every redraw, and the redraw that sent the
/// last one: the window's say, shared with the widget that ticks.
#[derive(Clone, Debug, Default)]
pub struct Pacer(Arc<Mutex<Ticks>>);

#[derive(Debug, Default)]
struct Ticks {
    every_redraw: bool,
    ticked: Option<Instant>,
}

/// What a redraw at the refresh does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    /// A tick for it, and the next redraw asked for.
    Tick,
    /// The next redraw asked for, no tick: this one was ticked already.
    Redraw,
    /// A redraw asked for at this instant, no tick: the last came too
    /// recently.
    RedrawAt(Instant),
}

impl Ticks {
    /// What the redraw at `at` does, ticking every redraw.
    fn redraw(&mut self, at: Instant) -> Option<Next> {
        if !self.every_redraw {
            return None;
        }
        Some(match self.ticked {
            // Once a redraw: iced hands the redraw to the widgets again
            // after the tick's update rebuilt them.
            Some(last) if last == at => Next::Redraw,
            // Redraws that do not wait for the display (a window going in
            // or out of fullscreen came thousands a second): at most one
            // tick every FASTEST.
            Some(last) if at.saturating_duration_since(last) < FASTEST => {
                Next::RedrawAt(last + FASTEST)
            }
            _ => {
                self.ticked = Some(at);
                Next::Tick
            }
        })
    }
}

impl Pacer {
    fn lock(&self) -> MutexGuard<'_, Ticks> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Ticks at the display's refresh (`on`) or not.
    pub fn tick_every_redraw(&self, on: bool) {
        self.lock().every_redraw = on;
    }

    /// `root`, ticked at the refresh while the window says so.
    pub fn view<'a>(&self, root: Element<'a, Message>) -> Element<'a, Message> {
        Paced {
            root,
            pacer: self.clone(),
        }
        .boxed()
    }
}

/// The widget: the root as it is (its tree, its layout, its drawing), the
/// redraws looked at first.
struct Paced<'a> {
    root: Element<'a, Message>,
    pacer: Pacer,
}

impl widget::Meta for Paced<'_> {
    fn is_void(&self) -> bool {
        self.root.is_void()
    }
}

impl Widget<Message, Theme, Renderer> for Paced<'_> {
    fn size(&self) -> Size<Length> {
        self.root.size()
    }

    fn tag(&self) -> tree::Tag {
        self.root.tag()
    }

    fn state(&self) -> tree::State {
        self.root.state()
    }

    fn diff(&mut self, tree: &mut Tree) {
        self.root.diff(tree);
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        self.root.layout(tree, renderer, limits);
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
        self.root
            .draw(tree, renderer, theme, style, layout, cursor, viewport);
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout,
        viewport: &Rectangle,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.root
            .operate(tree, layout, viewport, renderer, operation);
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
        if let Event::Window(iced::window::Event::RedrawRequested(at)) = event {
            match self.pacer.lock().redraw(*at) {
                Some(Next::Tick) => {
                    shell.publish(Message::Frame(*at));
                    shell.request_redraw();
                }
                Some(Next::Redraw) => shell.request_redraw(),
                Some(Next::RedrawAt(when)) => shell.request_redraw_at(when),
                None => {}
            }
        }
        self.root
            .update(tree, event, layout, cursor, renderer, shell, viewport);
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.root
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
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
        self.root
            .overlay(tree, layout, renderer, viewport, translation, window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticking every redraw: one tick a redraw, none for the same redraw
    /// handed over again (the next asked for), none for one that comes
    /// within [`FASTEST`] of the last tick (a redraw asked for then);
    /// not ticking every redraw, the redraws do nothing.
    #[test]
    fn a_tick_comes_once_a_redraw_and_at_most_every_fastest() {
        let t0 = Instant::now();
        let mut ticks = Ticks::default();
        assert_eq!(ticks.redraw(t0), None);
        ticks.every_redraw = true;
        assert_eq!(ticks.redraw(t0), Some(Next::Tick));
        assert_eq!(ticks.redraw(t0), Some(Next::Redraw));
        let soon = t0 + Duration::from_millis(1);
        assert_eq!(ticks.redraw(soon), Some(Next::RedrawAt(t0 + FASTEST)));
        assert_eq!(ticks.redraw(t0 + FASTEST), Some(Next::Tick));
        let refresh = t0 + FASTEST + Duration::from_micros(8_333);
        assert_eq!(ticks.redraw(refresh), Some(Next::Tick));
        ticks.every_redraw = false;
        assert_eq!(ticks.redraw(refresh + FASTEST), None);
    }
}
