//! A piano keyboard ([`piano`]): octaves of white keys with the black ones
//! over them, the keys held lit in the accent, a key's name or the
//! computer key that plays it written on it. It draws only: an
//! [`super::area`] over it takes the pointer, and [`key_at`] says which
//! key a point is on (a press there, a drag across the keys).

use iced::alignment::Vertical;
use iced::mouse::Cursor;
use iced::widget::canvas::{self, Frame, Geometry, Path, Program, Text};
use iced::widget::text::Alignment;
use iced::{Color, Element, Fill, Length, Point, Rectangle, Renderer, Size, Theme, Widget as _};

use crate::gui::look::{self, size};

/// The semitones of an octave that are white keys, from C.
const WHITE: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Black keys' share of a white key's width, and of the keys' height.
const BLACK_WIDTH: f32 = 0.6;
const BLACK_HEIGHT: f32 = 0.62;
/// The gap between white keys, in points.
const GAP: f32 = 2.0;

/// A keyboard of `octaves` from a C.
pub struct Piano {
    octaves: u8,
    /// The keys held, as semitones over the lowest C.
    lit: Vec<u8>,
    /// What is written on keys: a semitone over the lowest C and its word.
    labels: Vec<(u8, String)>,
}

/// A keyboard of `octaves`, the keys `lit` held, `labels` written on keys.
pub fn piano<'a, Message: 'a>(
    octaves: u8,
    lit: Vec<u8>,
    labels: Vec<(u8, String)>,
) -> Element<'a, Message> {
    canvas::Canvas::new(Piano {
        octaves: octaves.max(1),
        lit,
        labels,
    })
    .width(Fill)
    .height(Length::Fill)
    .boxed()
}

/// Whether semitone `k` (over a C) is a black key.
fn black(k: u8) -> bool {
    !WHITE.contains(&(k % 12))
}

/// The white keys before semitone `k` (over the lowest C).
fn whites_before(k: u8) -> u8 {
    (k / 12) * 7 + WHITE.iter().filter(|&&w| w < k % 12).count() as u8
}

/// Key `k`'s rectangle in a keyboard of `octaves` filling `size`.
fn rect_of(k: u8, octaves: u8, size: Size) -> Rectangle {
    let white = size.width / f32::from(7 * octaves);
    let x = f32::from(whites_before(k)) * white;
    if black(k) {
        let w = white * BLACK_WIDTH;
        Rectangle::new(
            Point::new(x - w / 2.0, 0.0),
            Size::new(w, size.height * BLACK_HEIGHT),
        )
    } else {
        Rectangle::new(Point::new(x, 0.0), Size::new(white, size.height))
    }
}

/// The key (a semitone over the lowest C) at `at` on a keyboard of
/// `octaves` in `bounds` (both in the window), the black keys over the
/// white; none off it.
pub fn key_at(bounds: Rectangle, octaves: u8, at: Point) -> Option<u8> {
    if !bounds.contains(at) || octaves == 0 {
        return None;
    }
    let p = Point::new(at.x - bounds.x, at.y - bounds.y);
    let keys = 12 * octaves;
    let size = bounds.size();
    (0..keys)
        .filter(|&k| black(k))
        .chain((0..keys).filter(|&k| !black(k)))
        .find(|&k| rect_of(k, octaves, size).contains(p))
}

impl<Message> Program<Message> for Piano {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let keys = 12 * self.octaves;
        let size = bounds.size();
        let lit = |k: u8| self.lit.contains(&k);
        let key = |frame: &mut Frame, k: u8, fill: Color| {
            let r = rect_of(k, self.octaves, size);
            let r = if black(k) {
                r
            } else {
                Rectangle::new(r.position(), Size::new(r.width - GAP, r.height))
            };
            frame.fill(
                &Path::rounded_rectangle(r.position(), r.size(), look::radius::CONTROL.into()),
                fill,
            );
        };
        for k in (0..keys).filter(|&k| !black(k)) {
            let fill = if lit(k) {
                look::ACCENT
            } else {
                look::tint(look::TEXT, look::RAISED, 0.85)
            };
            key(&mut frame, k, fill);
        }
        for k in (0..keys).filter(|&k| black(k)) {
            let fill = if lit(k) { look::ACCENT } else { look::BG };
            key(&mut frame, k, fill);
        }
        for (k, word) in &self.labels {
            if *k >= keys {
                continue;
            }
            let r = rect_of(*k, self.octaves, size);
            let ink = match (black(*k), lit(*k)) {
                (_, true) => look::ACCENT_INK,
                (true, false) => look::TEXT_DIM,
                (false, false) => look::alpha(look::BG, 0.7),
            };
            frame.fill_text(Text {
                content: word.clone(),
                position: Point::new(r.center_x(), r.y + r.height - 8.0),
                color: ink,
                size: size::CAPTION.into(),
                font: look::font::label(),
                align_x: Alignment::Center,
                align_y: Vertical::Bottom,
                ..Text::default()
            });
        }
        vec![frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_on_the_key_drawn_there() {
        let bounds = Rectangle::new(Point::new(100.0, 50.0), Size::new(700.0, 100.0));
        let white = 700.0 / 14.0;
        // Low on the keys: the white ones, C D E … across.
        let low = |x: f32| key_at(bounds, 2, Point::new(100.0 + x, 140.0));
        assert_eq!(low(white * 0.5), Some(0));
        assert_eq!(low(white * 1.5), Some(2));
        assert_eq!(low(white * 7.5), Some(12));
        // High between C and D: C sharp, over the white keys.
        assert_eq!(key_at(bounds, 2, Point::new(100.0 + white, 60.0)), Some(1));
        assert_eq!(
            key_at(bounds, 2, Point::new(100.0 + white * 3.0, 60.0)),
            Some(5)
        );
        assert_eq!(key_at(bounds, 2, Point::new(50.0, 60.0)), None);
        for k in 0..24 {
            let c = rect_of(k, 2, bounds.size()).center();
            let at = Point::new(c.x + bounds.x, c.y + bounds.y);
            assert_eq!(key_at(bounds, 2, at), Some(k), "{k}");
        }
    }
}
