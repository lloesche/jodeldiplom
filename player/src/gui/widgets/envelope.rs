//! An envelope's curve ([`envelope`]): an ADSR drawn as the synth plays it
//! ([`Adsr::level_at`]: a linear rise, an exponential fall to the sustain
//! level, held a while, an exponential release), the times to scale with
//! each other; a faint fill under the line in the accent.

use iced::mouse::Cursor;
use iced::widget::canvas::{self, path, Frame, Geometry, Path, Program, Stroke};
use iced::{Element, Fill, Length, Point, Rectangle, Renderer, Theme, Widget as _};
use jodel_synth::patch::Adsr;

use crate::gui::look;

/// The share of the width the held part takes (a note held as long as
/// its attack and decay, at least).
const HOLD: f32 = 0.18;
/// The points along each curved part.
const STEPS: usize = 24;

/// The curve of an ADSR.
pub struct Envelope {
    adsr: Adsr,
}

/// The curve of `adsr`.
pub fn envelope<'a, Message: 'a>(adsr: Adsr) -> Element<'a, Message> {
    canvas::Canvas::new(Envelope { adsr })
        .width(Fill)
        .height(Length::Fill)
        .boxed()
}

impl Envelope {
    /// The curve's points, x 0 to 1 across, y 0 (silent) to 1 (full): the
    /// envelope's level at the ends of its parts and along the curved
    /// ones, a struck one (no sustain) to its decay's end, a held one held
    /// a while then let go.
    fn points(&self) -> Vec<(f32, f32)> {
        let a = self.adsr;
        let (attack, decay) = (a.attack.max(0.0), a.decay.max(0.0));
        let held = a.sustain > 0.0;
        let release = if held { a.release.max(0.0) } else { 0.0 };
        let hold = if held {
            (attack + decay).max(0.05) * HOLD / (1.0 - HOLD)
        } else {
            0.0
        };
        let total = (attack + decay + hold + release).max(1e-3);
        let gate = attack + decay + hold;
        let along = |from: f32, length: f32| {
            (1..=STEPS).map(move |k| from + length * k as f32 / STEPS as f32)
        };
        let mut times = vec![0.0, attack];
        times.extend(along(attack, decay));
        if held {
            times.push(gate);
            times.extend(along(gate, release));
        }
        let mut p = vec![(0.0, 0.0)];
        p.extend(
            times[1..]
                .iter()
                .map(|&t| (t / total, a.level_at(t, gate).clamp(0.0, 1.0))),
        );
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
        let adsr = Adsr {
            attack: 0.1,
            decay: 0.3,
            sustain: 0.5,
            release: 0.4,
        };
        let p = Envelope { adsr }.points();
        assert_eq!(p[1].1, 1.0);
        assert!(p
            .iter()
            .all(|&(x, y)| (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)));
        assert!((p[STEPS + 1].1 - 0.5).abs() < 1e-3, "at the sustain");
        assert!(p.last().unwrap().1 < 1e-3 && (p.last().unwrap().0 - 1.0).abs() < 1e-4);
        let struck = Envelope {
            adsr: Adsr {
                sustain: 0.0,
                ..adsr
            },
        };
        let q = struck.points();
        assert_eq!(q.len(), STEPS + 2);
        assert!((q.last().unwrap().0 - 1.0).abs() < 1e-4);
    }
}
