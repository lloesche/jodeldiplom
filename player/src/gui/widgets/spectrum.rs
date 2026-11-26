//! The spectrum ([`spectrum`]): the model's plot ([`Plot`], laid out by
//! [`plot`] inside the widget's margins) drawn on a canvas, which both of
//! iced's renderers draw. The grid as faint hairlines, the axis labels in
//! tabular figures (kept between frames while the size and the rate
//! stay), and the curve as a line in the source's colour over a soft fill
//! fading down to nothing.

use std::cell::Cell;

use iced::advanced::mouse;
use iced::alignment::Vertical;
use iced::widget::canvas::{
    self, gradient, Cache, Frame, Geometry, LineCap, LineJoin, Path, Stroke, Text,
};
use iced::{
    Element, Fill, Padding, Pixels, Point, Rectangle, Renderer, Size, Theme, Vector, Widget as _,
};

use crate::gui::look::{self, size, space};
use crate::spectrum::SpectrumFrame;
use crate::view::plot::{LabelTone, Plot, Type};

/// The labels' type: the caption's size, its figures' width and its line.
pub const TYPE: Type = Type {
    figure: size::CAPTION * 0.6,
    line: size::CAPTION * 1.4,
};

/// The margins round the plot inside the widget, in points.
const INSET: Padding = Padding {
    top: space::S,
    right: space::L,
    bottom: space::XS,
    left: space::S,
};

/// The points across between two of the curve's points.
const STEP: f32 = 3.0;

/// The curve's line, in points.
const LINE: f32 = 1.5;

/// `frame` drawn to fill its place.
pub fn spectrum<'a, Message: 'a>(frame: &SpectrumFrame) -> Element<'a, Message> {
    canvas::Canvas::<Spectrum, Message>::new(Spectrum {
        frame: frame.clone(),
    })
    .width(Fill)
    .height(Fill)
    .boxed()
}

/// `frame` laid out for a widget of `size`: inside its margins, labelled
/// in [`TYPE`] (its coordinates from the margins' top left).
pub fn plot(size: Size, frame: &SpectrumFrame) -> Plot {
    Plot::new(
        size.width - INSET.left - INSET.right,
        size.height - INSET.top - INSET.bottom,
        frame,
        TYPE,
    )
}

/// The canvas's program: the spectrum to draw.
struct Spectrum {
    frame: SpectrumFrame,
}

/// What the program keeps between frames: the grid and the labels, and
/// what they were drawn for (the size and the rate).
#[derive(Default)]
struct Kept {
    grid: Cache,
    for_: Cell<Option<(Size, u32)>>,
}

impl<Message> canvas::Program<Message> for Spectrum {
    type State = Kept;

    fn draw(
        &self,
        kept: &Kept,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let size = bounds.size();
        let plot = plot(size, &self.frame);
        let key = Some((size, self.frame.rate));
        if kept.for_.get() != key {
            kept.grid.clear();
            kept.for_.set(key);
        }
        let inset = Vector::new(INSET.left, INSET.top);
        let grid = kept.grid.draw(renderer, size, |frame| {
            frame.translate(inset);
            grid(frame, &plot);
        });
        let mut curve = Frame::new(renderer, size);
        draw_curve(&mut curve, &plot, inset);
        vec![grid, curve.into_geometry()]
    }
}

/// The grid's lines and the axis labels.
fn grid(frame: &mut Frame, plot: &Plot) {
    let area = plot.area;
    let hair = Stroke::default()
        .with_width(1.0)
        .with_color(look::tint(look::TEXT, look::BG, 0.07));
    for &db in &plot.db_lines {
        // On a whole point, so the hairline is sharp.
        let y = plot.y_of(db).round() + 0.5;
        frame.stroke(
            &Path::line(Point::new(area.x, y), Point::new(area.x + area.width, y)),
            hair,
        );
    }
    for &hz in &plot.hz_lines {
        let x = plot.x_of(hz).round() + 0.5;
        frame.stroke(
            &Path::line(Point::new(x, area.y), Point::new(x, area.y + area.height)),
            hair,
        );
    }
    for label in plot.db_labels.iter().chain(&plot.hz_labels) {
        frame.fill_text(Text {
            content: label.text.clone(),
            position: Point::new(label.x, label.y),
            color: match label.tone {
                LabelTone::Dim => look::TEXT_FAINT,
                LabelTone::Source => look::SOURCE,
            },
            size: Pixels(size::CAPTION),
            font: look::font::numbers(),
            align_y: Vertical::Center,
            ..Text::default()
        });
    }
}

/// The curve, the plot's origin at `origin`: a soft fill down to the
/// floor, the line over it, clipped to the plot's area.
fn draw_curve(frame: &mut Frame, plot: &Plot, origin: Vector) {
    let points: Vec<(f32, f32)> = plot
        .curve(STEP)
        .into_iter()
        .map(|(x, y)| (x + origin.x, y + origin.y))
        .collect();
    let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
        return;
    };
    let area = plot.area;
    let top = area.y + origin.y;
    let floor = top + area.height;
    // Silence lies a line below the floor, out of sight.
    let at = |(x, y): (f32, f32)| Point::new(x, if y >= floor - 0.25 { floor + LINE } else { y });
    let line = Path::new(|b| {
        b.move_to(at(first));
        for &p in &points[1..] {
            b.line_to(at(p));
        }
    });
    let under = Path::new(|b| {
        b.move_to(Point::new(first.0, floor + LINE));
        for &p in &points {
            b.line_to(at(p));
        }
        b.line_to(Point::new(last.0, floor + LINE));
        b.close();
    });
    let fade = gradient::Linear::new(Point::new(0.0, top), Point::new(0.0, floor))
        .add_stop(0.0, look::alpha(look::SOURCE, 0.16))
        .add_stop(1.0, look::alpha(look::SOURCE, 0.0));
    frame.with_clip(
        Rectangle::new(
            Point::new(area.x + origin.x, top),
            Size::new(area.width, area.height),
        ),
        |frame| {
            frame.fill(&under, fade);
            frame.stroke(
                &line,
                Stroke {
                    width: LINE,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    ..Stroke::default()
                }
                .with_color(look::SOURCE),
            );
        },
    );
}
