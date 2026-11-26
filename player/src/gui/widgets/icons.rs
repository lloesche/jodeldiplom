//! The interface's icons, drawn as vector paths: one stroke weight (two
//! units of a 24-unit square, scaled with the icon), round ends, sized
//! from the type scale. Each draws in the colour of the text around it (a
//! button's, as it eases between its states) unless it is given its own.
//! No icon fonts and no emoji for the chrome.

use iced::advanced::graphics::geometry::Renderer as _;
use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{Tree, Widget};
use iced::advanced::{mouse, Renderer as _};
use iced::widget::canvas::{path, Frame, LineCap, LineJoin, Path, Stroke};
use iced::{Color, Length, Point, Rectangle, Renderer, Size, Theme, Vector};

use crate::gui::look;

/// An icon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    /// Done.
    Check,
    /// Failed; close.
    Cross,
    /// Try again.
    Retry,
    /// Getting files.
    Download,
    /// Something needs attention.
    Alert,
}

impl Icon {
    /// Every icon, for the look sheet.
    pub const ALL: [Icon; 5] = [
        Icon::Check,
        Icon::Cross,
        Icon::Retry,
        Icon::Download,
        Icon::Alert,
    ];

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Icon::Check => "check",
            Icon::Cross => "cross",
            Icon::Retry => "retry",
            Icon::Download => "download",
            Icon::Alert => "alert",
        }
    }
}

/// `icon` as a widget, [`look::height::ICON`] square.
pub fn icon(icon: Icon) -> Glyph {
    Glyph {
        icon,
        size: look::height::ICON,
        color: None,
    }
}

/// An icon's widget.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    icon: Icon,
    size: f32,
    color: Option<Color>,
}

impl Glyph {
    /// `size` points square.
    pub fn size(mut self, size: f32) -> Glyph {
        self.size = size;
        self
    }

    /// In `color`, not the text's.
    pub fn color(mut self, color: Color) -> Glyph {
        self.color = Some(color);
        self
    }
}

impl iced::advanced::widget::Meta for Glyph {}

impl<Message> Widget<Message, Theme, Renderer> for Glyph {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(self.size), Length::Fixed(self.size))
    }

    fn layout(&mut self, tree: &mut Tree, _renderer: &Renderer, limits: &layout::Limits) {
        let size = limits.resolve(self.size, self.size, Size::new(self.size, self.size));
        tree.size = size;
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        if bounds.intersection(viewport).is_none() {
            return;
        }
        let mut frame = Frame::new(renderer, bounds.size());
        paint(
            &mut frame,
            self.icon,
            bounds.width,
            self.color.unwrap_or(style.text_color),
        );
        renderer.with_translation(Vector::new(bounds.x, bounds.y), |renderer| {
            renderer.draw_geometry(frame.into_geometry());
        });
    }
}

/// Draws `icon` into `frame`, `size` points square, in `color`.
fn paint(frame: &mut Frame, icon: Icon, size: f32, color: Color) {
    // The icons are drawn on a 24-unit square.
    let k = size / 24.0;
    let p = |x: f32, y: f32| Point::new(x * k, y * k);
    let stroke = Stroke {
        width: 2.0 * k,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    }
    .with_color(color);
    let lines = |points: &[(f32, f32)]| {
        Path::new(|b| {
            for (i, &(x, y)) in points.iter().enumerate() {
                if i == 0 {
                    b.move_to(p(x, y));
                } else {
                    b.line_to(p(x, y));
                }
            }
        })
    };
    match icon {
        Icon::Check => frame.stroke(&lines(&[(5.0, 12.5), (10.0, 17.5), (19.0, 7.0)]), stroke),
        Icon::Cross => {
            frame.stroke(&lines(&[(6.0, 6.0), (18.0, 18.0)]), stroke);
            frame.stroke(&lines(&[(18.0, 6.0), (6.0, 18.0)]), stroke);
        }
        Icon::Retry => {
            // Most of a circle, turning clockwise, an arrow's head at its
            // end near the top.
            let (cx, cy, r) = (12.0, 12.5, 7.5);
            let start = -50f32.to_radians();
            let end = start + 290f32.to_radians();
            frame.stroke(
                &Path::new(|b| {
                    b.arc(path::Arc {
                        center: p(cx, cy),
                        radius: r * k,
                        start_angle: iced::Radians(start),
                        end_angle: iced::Radians(end),
                    });
                }),
                stroke,
            );
            let (x, y) = (cx + r * end.cos(), cy + r * end.sin());
            let (dx, dy) = (-end.sin(), end.cos());
            let head = |turn: f32| {
                let (s, c) = turn.to_radians().sin_cos();
                let (bx, by) = (-(dx * c - dy * s), -(dx * s + dy * c));
                (x + bx * 4.5, y + by * 4.5)
            };
            frame.stroke(&lines(&[head(40.0), (x, y), head(-40.0)]), stroke);
        }
        Icon::Download => {
            frame.stroke(&lines(&[(12.0, 3.5), (12.0, 15.0)]), stroke);
            frame.stroke(&lines(&[(7.0, 10.0), (12.0, 15.0), (17.0, 10.0)]), stroke);
            frame.stroke(
                &lines(&[(4.0, 15.5), (4.0, 20.0), (20.0, 20.0), (20.0, 15.5)]),
                stroke,
            );
        }
        Icon::Alert => {
            frame.stroke(
                &Path::new(|b| {
                    b.move_to(p(12.0, 3.5));
                    b.line_to(p(21.5, 20.0));
                    b.line_to(p(2.5, 20.0));
                    b.close();
                }),
                stroke,
            );
            frame.stroke(&lines(&[(12.0, 9.5), (12.0, 13.5)]), stroke);
            frame.fill(&Path::circle(p(12.0, 16.9), 1.25 * k), color);
        }
    }
}
