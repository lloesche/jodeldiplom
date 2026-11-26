//! The interface's icons, drawn as vector paths: one stroke weight (two
//! units of a 24-unit square, scaled with the icon), round ends, sized
//! from the type scale; the transport's shapes are filled inside that
//! stroke, so their corners are as round. Each draws in the colour of the
//! text around it (a button's, as it eases between its states) unless it
//! is given its own. No icon fonts and no emoji for the chrome.

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
    /// Play.
    Play,
    /// Pause.
    Pause,
    /// The track before.
    Previous,
    /// The next track.
    Next,
    /// Stop.
    Stop,
    /// The settings.
    Settings,
    /// The keys and what they do.
    Help,
    /// A folder.
    Folder,
    /// A music file.
    Note,
    /// A playlist.
    Playlist,
    /// The queue.
    Queue,
    /// Up a folder.
    Up,
    /// Into a folder.
    Chevron,
    /// Looking for something: the filter, a search.
    Search,
    /// A command typed.
    Command,
    /// A track's key (a sharp).
    Key,
    /// A track's stems.
    Stems,
    /// A track's tempo (a metronome).
    Tempo,
    /// A track's lyrics.
    Lyrics,
    /// A track's cover.
    Cover,
    /// The microphone.
    Mic,
    /// Set by hand, to stay where it is (a pushpin).
    Pin,
    /// Back a change.
    Undo,
    /// Forward a change undone.
    Redo,
    /// Work under way: an arc that turns ([`Glyph::turn`]).
    Busy,
}

impl Icon {
    /// Every icon, for the look sheet.
    pub const ALL: [Icon; 30] = [
        Icon::Check,
        Icon::Cross,
        Icon::Retry,
        Icon::Download,
        Icon::Alert,
        Icon::Play,
        Icon::Pause,
        Icon::Previous,
        Icon::Next,
        Icon::Stop,
        Icon::Settings,
        Icon::Help,
        Icon::Folder,
        Icon::Note,
        Icon::Playlist,
        Icon::Queue,
        Icon::Up,
        Icon::Chevron,
        Icon::Search,
        Icon::Command,
        Icon::Key,
        Icon::Stems,
        Icon::Tempo,
        Icon::Lyrics,
        Icon::Cover,
        Icon::Mic,
        Icon::Pin,
        Icon::Undo,
        Icon::Redo,
        Icon::Busy,
    ];

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Icon::Check => "check",
            Icon::Cross => "cross",
            Icon::Retry => "retry",
            Icon::Download => "download",
            Icon::Alert => "alert",
            Icon::Play => "play",
            Icon::Pause => "pause",
            Icon::Previous => "previous",
            Icon::Next => "next",
            Icon::Stop => "stop",
            Icon::Settings => "settings",
            Icon::Help => "help",
            Icon::Folder => "folder",
            Icon::Note => "note",
            Icon::Playlist => "playlist",
            Icon::Queue => "queue",
            Icon::Up => "up",
            Icon::Chevron => "chevron",
            Icon::Search => "search",
            Icon::Command => "command",
            Icon::Key => "key",
            Icon::Stems => "stems",
            Icon::Tempo => "tempo",
            Icon::Lyrics => "lyrics",
            Icon::Cover => "cover",
            Icon::Mic => "mic",
            Icon::Pin => "pin",
            Icon::Undo => "undo",
            Icon::Redo => "redo",
            Icon::Busy => "busy",
        }
    }
}

/// `icon` as a widget, [`look::height::ICON`] square.
pub fn icon(icon: Icon) -> Glyph {
    Glyph {
        icon,
        size: look::height::ICON,
        color: None,
        turn: 0.0,
    }
}

/// An icon's widget.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    icon: Icon,
    size: f32,
    color: Option<Color>,
    turn: f32,
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

    /// Turned by `turn` of a whole turn (0 to 1), clockwise: where the
    /// busy arc is ([`Icon::Busy`]).
    pub fn turn(mut self, turn: f32) -> Glyph {
        self.turn = turn;
        self
    }
}

impl iced::advanced::widget::Meta for Glyph {}

impl<'a, Message: 'a> From<Glyph> for iced::Element<'a, Message> {
    fn from(glyph: Glyph) -> Self {
        iced::Widget::boxed(glyph)
    }
}

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
            self.turn,
        );
        renderer.with_translation(Vector::new(bounds.x, bounds.y), |renderer| {
            renderer.draw_geometry(frame.into_geometry());
        });
    }
}

/// Draws `icon` into `frame`, `size` points square, in `color`, turned
/// by `turn` of a whole turn where it turns.
fn paint(frame: &mut Frame, icon: Icon, size: f32, color: Color, turn: f32) {
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
    // A shape filled, its outline in the stroke (round corners).
    let solid = |frame: &mut Frame, shape: &Path| {
        frame.fill(shape, color);
        frame.stroke(shape, stroke);
    };
    let polygon = |points: &[(f32, f32)]| {
        Path::new(|b| {
            for (i, &(x, y)) in points.iter().enumerate() {
                if i == 0 {
                    b.move_to(p(x, y));
                } else {
                    b.line_to(p(x, y));
                }
            }
            b.close();
        })
    };
    let bar = |x: f32, y: f32, w: f32, h: f32, r: f32| {
        Path::rounded_rectangle(p(x, y), Size::new(w * k, h * k), (r * k).into())
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
        Icon::Play => solid(frame, &polygon(&[(8.0, 5.5), (18.5, 12.0), (8.0, 18.5)])),
        Icon::Pause => {
            frame.fill(&bar(6.5, 5.0, 4.0, 14.0, 1.2), color);
            frame.fill(&bar(13.5, 5.0, 4.0, 14.0, 1.2), color);
        }
        Icon::Stop => frame.fill(&bar(6.0, 6.0, 12.0, 12.0, 2.0), color),
        Icon::Next => {
            solid(frame, &polygon(&[(6.0, 6.5), (14.5, 12.0), (6.0, 17.5)]));
            frame.stroke(&lines(&[(18.0, 6.0), (18.0, 18.0)]), stroke);
        }
        Icon::Previous => {
            solid(frame, &polygon(&[(18.0, 6.5), (9.5, 12.0), (18.0, 17.5)]));
            frame.stroke(&lines(&[(6.0, 6.0), (6.0, 18.0)]), stroke);
        }
        Icon::Settings => {
            // A cog: eight teeth round a ring, a hole in the middle.
            let (cx, cy) = (12.0, 12.0);
            let teeth = 8;
            let mut points = Vec::with_capacity(teeth * 4);
            for i in 0..teeth {
                let at = i as f32 / teeth as f32 * std::f32::consts::TAU;
                let half = std::f32::consts::TAU / teeth as f32 / 2.0;
                for (turn, r) in [
                    (-half, 6.6),
                    (-half * 0.45, 9.2),
                    (half * 0.45, 9.2),
                    (half, 6.6),
                ] {
                    let a = at + turn;
                    points.push((cx + r * a.cos(), cy + r * a.sin()));
                }
            }
            frame.stroke(&polygon(&points), stroke);
            frame.stroke(&Path::circle(p(cx, cy), 2.8 * k), stroke);
        }
        Icon::Help => {
            frame.stroke(&Path::circle(p(12.0, 12.0), 9.0 * k), stroke);
            // The hook over the top, down to the stem.
            let (cx, cy, r) = (12.0, 9.8, 2.6);
            let end = 0.95f32;
            frame.stroke(
                &Path::new(|b| {
                    b.arc(path::Arc {
                        center: p(cx, cy),
                        radius: r * k,
                        start_angle: iced::Radians(std::f32::consts::PI),
                        end_angle: iced::Radians(std::f32::consts::TAU + end),
                    });
                    b.line_to(p(12.0, 13.6));
                }),
                stroke,
            );
            frame.fill(&Path::circle(p(12.0, 16.9), 1.25 * k), color);
        }
        Icon::Folder => frame.stroke(
            &polygon(&[
                (3.0, 6.0),
                (9.5, 6.0),
                (11.5, 8.0),
                (21.0, 8.0),
                (21.0, 18.5),
                (3.0, 18.5),
            ]),
            stroke,
        ),
        Icon::Note => {
            frame.stroke(&lines(&[(11.5, 16.5), (11.5, 4.5), (17.5, 7.5)]), stroke);
            frame.fill(&Path::circle(p(8.6, 16.6), 3.0 * k), color);
        }
        Icon::Playlist => {
            frame.stroke(&lines(&[(3.5, 6.5), (14.5, 6.5)]), stroke);
            frame.stroke(&lines(&[(3.5, 11.5), (14.5, 11.5)]), stroke);
            frame.stroke(&lines(&[(3.5, 16.5), (9.5, 16.5)]), stroke);
            frame.stroke(&lines(&[(19.0, 17.0), (19.0, 9.0), (21.5, 10.0)]), stroke);
            frame.fill(&Path::circle(p(16.6, 17.2), 2.4 * k), color);
        }
        Icon::Queue => {
            frame.stroke(&lines(&[(3.5, 6.5), (20.5, 6.5)]), stroke);
            frame.stroke(&lines(&[(3.5, 11.5), (12.5, 11.5)]), stroke);
            frame.stroke(&lines(&[(3.5, 16.5), (12.5, 16.5)]), stroke);
            solid(frame, &polygon(&[(16.0, 11.0), (21.0, 14.0), (16.0, 17.0)]));
        }
        Icon::Up => {
            frame.stroke(&lines(&[(12.0, 19.5), (12.0, 5.0)]), stroke);
            frame.stroke(&lines(&[(6.5, 10.5), (12.0, 5.0), (17.5, 10.5)]), stroke);
        }
        Icon::Chevron => frame.stroke(&lines(&[(9.5, 6.0), (15.5, 12.0), (9.5, 18.0)]), stroke),
        Icon::Search => {
            frame.stroke(&Path::circle(p(10.5, 10.5), 6.0 * k), stroke);
            frame.stroke(&lines(&[(15.0, 15.0), (20.0, 20.0)]), stroke);
        }
        Icon::Command => {
            frame.stroke(&lines(&[(5.0, 7.0), (10.0, 12.0), (5.0, 17.0)]), stroke);
            frame.stroke(&lines(&[(13.0, 17.0), (19.0, 17.0)]), stroke);
        }
        Icon::Key => {
            frame.stroke(&lines(&[(10.5, 4.0), (9.0, 20.0)]), stroke);
            frame.stroke(&lines(&[(16.0, 4.0), (14.5, 20.0)]), stroke);
            frame.stroke(&lines(&[(5.0, 10.0), (20.0, 8.0)]), stroke);
            frame.stroke(&lines(&[(4.0, 16.0), (19.0, 14.0)]), stroke);
        }
        Icon::Stems => {
            for (x, half) in [(5.0, 3.0), (9.67, 7.0), (14.33, 4.5), (19.0, 6.0)] {
                frame.stroke(&lines(&[(x, 12.0 - half), (x, 12.0 + half)]), stroke);
            }
        }
        Icon::Tempo => {
            frame.stroke(
                &polygon(&[(7.5, 20.0), (10.5, 4.0), (13.5, 4.0), (16.5, 20.0)]),
                stroke,
            );
            frame.stroke(&lines(&[(12.0, 15.5), (17.5, 7.5)]), stroke);
        }
        Icon::Lyrics => {
            // A speech bubble, two lines of text in it.
            frame.stroke(&bar(3.5, 4.0, 17.0, 12.5, 3.5), stroke);
            frame.stroke(&lines(&[(8.0, 16.5), (7.0, 20.5), (12.0, 16.5)]), stroke);
            frame.stroke(&lines(&[(8.0, 8.5), (16.0, 8.5)]), stroke);
            frame.stroke(&lines(&[(8.0, 12.0), (13.0, 12.0)]), stroke);
        }
        Icon::Cover => {
            frame.stroke(&bar(3.5, 3.5, 17.0, 17.0, 3.5), stroke);
            frame.fill(&Path::circle(p(9.0, 9.0), 1.8 * k), color);
            frame.stroke(
                &lines(&[
                    (4.0, 17.5),
                    (10.0, 12.0),
                    (14.0, 15.5),
                    (16.5, 13.5),
                    (20.0, 16.5),
                ]),
                stroke,
            );
        }
        Icon::Mic => {
            // A capsule on a stand, a cradle round its lower half.
            frame.stroke(&bar(9.0, 3.0, 6.0, 11.5, 3.0), stroke);
            frame.stroke(
                &Path::new(|b| {
                    b.arc(path::Arc {
                        center: p(12.0, 11.0),
                        radius: 6.0 * k,
                        start_angle: iced::Radians(0.0),
                        end_angle: iced::Radians(std::f32::consts::PI),
                    });
                }),
                stroke,
            );
            frame.stroke(&lines(&[(12.0, 17.0), (12.0, 20.5)]), stroke);
            frame.stroke(&lines(&[(8.5, 20.5), (15.5, 20.5)]), stroke);
        }
        Icon::Pin => {
            // A pushpin's head and collar, its needle under them.
            frame.stroke(
                &polygon(&[
                    (9.0, 4.0),
                    (15.0, 4.0),
                    (14.0, 9.5),
                    (17.5, 13.5),
                    (6.5, 13.5),
                    (10.0, 9.5),
                ]),
                stroke,
            );
            frame.stroke(&lines(&[(12.0, 13.5), (12.0, 20.5)]), stroke);
        }
        Icon::Undo | Icon::Redo => {
            // An arrow back to the left (to the right: redo), round a turn
            // under it.
            let x = |v: f32| if icon == Icon::Undo { v } else { 24.0 - v };
            let a = |v: f32| {
                if icon == Icon::Undo {
                    v
                } else {
                    std::f32::consts::PI - v
                }
            };
            frame.stroke(
                &lines(&[(x(8.5), 5.0), (x(4.5), 9.5), (x(8.5), 14.0)]),
                stroke,
            );
            frame.stroke(&lines(&[(x(4.5), 9.5), (x(14.5), 9.5)]), stroke);
            frame.stroke(&lines(&[(x(14.5), 19.0), (x(9.5), 19.0)]), stroke);
            let (from, to) = (
                a(-std::f32::consts::FRAC_PI_2),
                a(std::f32::consts::FRAC_PI_2),
            );
            frame.stroke(
                &Path::new(|b| {
                    b.arc(path::Arc {
                        center: p(x(14.5), 14.25),
                        radius: 4.75 * k,
                        start_angle: iced::Radians(from.min(to)),
                        end_angle: iced::Radians(from.max(to)),
                    });
                }),
                stroke,
            );
        }
        Icon::Busy => {
            let start = turn * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            frame.stroke(
                &Path::new(|b| {
                    b.arc(path::Arc {
                        center: p(12.0, 12.0),
                        radius: 8.0 * k,
                        start_angle: iced::Radians(start),
                        end_angle: iced::Radians(start + 1.5 * std::f32::consts::PI),
                    });
                }),
                stroke,
            );
        }
    }
}
