//! An envelope's curve ([`envelope`]): an ADSR drawn as it sounds, a
//! linear rise, an exponential fall to the sustain level (to within 60 dB
//! of it), held a while, and an exponential release, the times to scale
//! with each other; a faint fill under the line in the accent.

use iced::mouse::Cursor;
use iced::widget::canvas::{self, path, Frame, Geometry, Path, Program, Stroke};
use iced::{Element, Fill, Length, Point, Rectangle, Renderer, Theme, Widget as _};

use crate::gui::look;

/// The share of the width the held part takes (a note held as long as
/// its attack and decay, at least).
const HOLD: f32 = 0.18;
/// The points along each curved part.
const STEPS: usize = 24;
/// What an exponential part falls by over its time: 60 dB.
const FALL: f32 = 6.907_755;

/// The curve of an ADSR: its attack, decay and release in seconds, its
/// sustain level 0 to 1.
pub struct Envelope {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
}

/// The curve of `attack`, `decay` and `release` seconds and `sustain`.
pub fn envelope<'a, Message: 'a>(
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
) -> Element<'a, Message> {
    canvas::Canvas::new(Envelope {
        attack: attack.max(0.0),
        decay: decay.max(0.0),
        sustain: sustain.clamp(0.0, 1.0),
        release: release.max(0.0),
    })
    .width(Fill)
    .height(Length::Fill)
    .boxed()
}

impl Envelope {
    /// The curve's points, x 0 to 1 across, y 0 (silent) to 1 (full).
    fn points(&self) -> Vec<(f32, f32)> {
        let s = self.sustain;
        // A struck sound rings out in its decay; a held one holds, then is
        // let go.
        let held = s > 0.0;
        let release = if held { self.release } else { 0.0 };
        let hold = if held {
            (self.attack + self.decay).max(0.05) * HOLD / (1.0 - HOLD)
        } else {
            0.0
        };
        let total = (self.attack + self.decay + hold + release).max(1e-3);
        let x = |t: f32| t / total;
        let mut p = vec![(0.0, 0.0), (x(self.attack), 1.0)];
        for k in 1..=STEPS {
            let u = k as f32 / STEPS as f32;
            let level = s + (1.0 - s) * (-FALL * u).exp();
            p.push((x(self.attack + u * self.decay), level));
        }
        if held {
            let end = self.attack + self.decay + hold;
            p.push((x(end), s));
            for k in 1..=STEPS {
                let u = k as f32 / STEPS as f32;
                p.push((x(end + u * release), s * (-FALL * u).exp()));
            }
        }
        p
    }
}

impl<Message> Program<Message> for Envelope {
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
        let (w, h) = (bounds.width, bounds.height);
        let pad = 3.0;
        let at = |(x, y): (f32, f32)| {
            Point::new(pad + x * (w - 2.0 * pad), h - pad - y * (h - 2.0 * pad))
        };
        let points = self.points();
        let line = Path::new(|b: &mut path::Builder| {
            b.move_to(at(points[0]));
            for &p in &points[1..] {
                b.line_to(at(p));
            }
        });
        let fill = Path::new(|b: &mut path::Builder| {
            b.move_to(at((0.0, 0.0)));
            for &p in &points {
                b.line_to(at(p));
            }
            b.line_to(at((points.last().map_or(1.0, |p| p.0), 0.0)));
            b.close();
        });
        frame.fill(&fill, look::alpha(look::ACCENT, 0.12));
        frame.stroke(
            &line,
            Stroke::default().with_color(look::ACCENT).with_width(2.0),
        );
        vec![frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The curve rises, falls to its sustain and is let go; a struck one
    /// ends in its decay; every point within the box.
    #[test]
    fn the_curve_goes_as_the_envelope_does() {
        let held = Envelope {
            attack: 0.1,
            decay: 0.3,
            sustain: 0.5,
            release: 0.4,
        };
        let p = held.points();
        assert_eq!(p[1].1, 1.0);
        assert!(p
            .iter()
            .all(|&(x, y)| (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)));
        assert!((p[STEPS + 1].1 - 0.5).abs() < 1e-3, "at the sustain");
        assert!(p.last().unwrap().1 < 1e-3 && (p.last().unwrap().0 - 1.0).abs() < 1e-4);
        let struck = Envelope {
            sustain: 0.0,
            ..held
        };
        let q = struck.points();
        assert_eq!(q.len(), STEPS + 2);
        assert!((q.last().unwrap().0 - 1.0).abs() < 1e-4);
    }
}
