//! A waveform ([`wave`]): bars of its loudness across, mirrored about
//! the middle, in a tone up to the share filled and faint past it; ticks
//! over it (a bar's first taller, in the accent), a play head, and the tick
//! it last passed lit by a pulse. It draws only, and takes no pointer.

use iced::mouse::Cursor;
use iced::widget::canvas::{self, Frame, Geometry, Path, Program, Stroke};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Theme, Widget as _};

use crate::gui::look;

/// A bar's share of its slot (the rest the gap).
const BAR: f32 = 0.62;
/// The least a bar is drawn at, in points (silence still shows).
const LEAST: f32 = 1.0;

/// A waveform to draw.
pub struct Wave {
    peaks: Vec<f32>,
    tone: Color,
    filled: f32,
    ticks: Vec<(f32, bool)>,
    head: Option<f32>,
    pulse: (f32, bool),
}

/// The waveform of `peaks` (each 0 to 1, left to right) in `tone`.
pub fn wave(peaks: Vec<f32>, tone: Color) -> Wave {
    Wave {
        peaks,
        tone,
        filled: 1.0,
        ticks: Vec::new(),
        head: None,
        pulse: (0.0, false),
    }
}

impl Wave {
    /// Only `share` of it (0 to 1, from the left) in its tone, the rest
    /// faint: what is not there yet.
    pub fn filled(mut self, share: f32) -> Wave {
        self.filled = share.clamp(0.0, 1.0);
        self
    }

    /// Ticks over it: where across (0 to 1), and whether each is a bar's
    /// first.
    pub fn ticks(mut self, ticks: Vec<(f32, bool)>) -> Wave {
        self.ticks = ticks;
        self
    }

    /// A play head across it (0 to 1), the tick it last passed lit by
    /// `pulse` (0 to 1).
    pub fn head(mut self, head: Option<f32>, pulse: (f32, bool)) -> Wave {
        self.head = head;
        self.pulse = pulse;
        self
    }

    /// It `height` points high, as wide as there is room.
    pub fn view<'a, Message: 'a>(self, height: f32) -> Element<'a, Message> {
        canvas::Canvas::new(self)
            .width(Length::Fill)
            .height(height)
            .boxed()
    }
}

impl<Message> Program<Message> for Wave {
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
        let n = self.peaks.len().max(1) as f32;
        let slot = w / n;
        let faint = look::alpha(look::TEXT_FAINT, 0.35);
        for (k, &p) in self.peaks.iter().enumerate() {
            let x = k as f32 * slot;
            let tall = (p.clamp(0.0, 1.0) * h).max(LEAST);
            let across = (k as f32 + 0.5) / n;
            let tone = if across <= self.filled {
                self.tone
            } else {
                faint
            };
            frame.fill_rectangle(
                Point::new(x + slot * (1.0 - BAR) / 2.0, (h - tall) / 2.0),
                Size::new((slot * BAR).max(1.0), tall),
                tone,
            );
        }
        // The tick the head passed last, lit by the pulse.
        let lit = self.head.and_then(|h| {
            (self.ticks.iter())
                .rposition(|(x, _)| *x <= h)
                .filter(|_| self.pulse.0 > 0.0)
        });
        for (k, &(x, bar)) in self.ticks.iter().enumerate() {
            let (reach, tone) = if bar {
                (1.0, look::ACCENT)
            } else {
                (0.55, look::alpha(look::TEXT, 0.55))
            };
            let glow = if lit == Some(k) { self.pulse.0 } else { 0.0 };
            let tone = crate::gui::style::mix(tone, look::TEXT, glow);
            let top = h * (1.0 - reach) / 2.0;
            let line = Path::line(Point::new(x * w, top), Point::new(x * w, h - top));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(tone)
                    .with_width(1.5 + 2.0 * glow),
            );
        }
        if let Some(head) = self.head {
            let line = Path::line(Point::new(head * w, 0.0), Point::new(head * w, h));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(look::alpha(look::TEXT, 0.8))
                    .with_width(1.0),
            );
        }
        vec![frame.into_geometry()]
    }
}
