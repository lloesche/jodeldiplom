//! Keeping what the keys are on in view: an operation ([`Reveal`]) that
//! scrolls a scrollable, where it must, so the widget found by an id
//! inside it shows whole (a little room round it), and leaves it where it
//! is otherwise.
//!
//! It looks first (where the scrollable is and how far it is scrolled,
//! where the widget is in what it scrolls), then scrolls in a second pass
//! ([`Outcome::Chain`]): the widget's place is known only once the
//! scrollable has been passed.

use iced::advanced::widget::operation::scrollable::{self, AbsoluteOffset, Scrollable};
use iced::advanced::widget::operation::{Animation, Outcome};
use iced::advanced::widget::{Id, Operation};
use iced::{Rectangle, Size, Vector};

/// The room kept above and below the widget, in points.
const ROOM: f32 = 12.0;

/// Scrolls the scrollable `scroller` so the widget `target` in it shows.
pub struct Reveal {
    scroller: Id,
    target: Id,
    /// The scrollable's bounds and how far it is scrolled.
    seen: Option<(Rectangle, Vector)>,
    /// Where what it scrolls starts.
    content: Option<f32>,
    /// Where the widget is, unscrolled.
    found: Option<Rectangle>,
}

impl Reveal {
    /// The operation keeping `target` in view in `scroller`.
    pub fn new(scroller: impl Into<Id>, target: impl Into<Id>) -> Reveal {
        Reveal {
            scroller: scroller.into(),
            target: target.into(),
            seen: None,
            content: None,
            found: None,
        }
    }

    /// The offset to scroll to (none: it shows already, or was not found).
    fn offset(&self) -> Option<f32> {
        let ((bounds, scrolled), top, found) = (self.seen?, self.content?, self.found?);
        let (from, to) = (found.y - top - ROOM, found.y - top + found.height + ROOM);
        if from < scrolled.y {
            Some(from.max(0.0))
        } else if to > scrolled.y + bounds.height {
            Some(to - bounds.height)
        } else {
            None
        }
    }
}

impl<T: 'static> Operation<T> for Reveal {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation<T>)) {
        operate(self);
    }

    fn scrollable(
        &mut self,
        id: Option<&Id>,
        bounds: Rectangle,
        _content: Size,
        translation: Vector,
        _state: &mut dyn Scrollable,
    ) {
        if id == Some(&self.scroller) {
            self.seen = Some((bounds, translation));
        }
    }

    fn container(&mut self, id: Option<&Id>, bounds: Rectangle, _viewport: &Rectangle) {
        if id == Some(&self.scroller) {
            self.content.get_or_insert(bounds.y);
        } else if id == Some(&self.target) {
            self.found = Some(bounds);
        }
    }

    fn finish(&self) -> Outcome<T> {
        match self.offset() {
            Some(y) => Outcome::Chain(Box::new(scrollable::scroll_to(
                self.scroller.clone(),
                AbsoluteOffset {
                    x: None,
                    y: Some(y),
                },
                Animation::Instant,
            ))),
            None => Outcome::None,
        }
    }
}
