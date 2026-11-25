//! A strip's tall section as a canvas: the scale, the fader's slot and its
//! cap, and the meters with their peak-hold ticks and clip lamps. What
//! stands still (the scale, the slot, the meters' unlit faces) is drawn
//! once into a cache, again only when the size changes; the cap and the
//! lit bars every frame, a few rectangles.
//!
//! The mouse: a press on the cap takes it where it is, elsewhere on the
//! slot brings it to the pointer; held, it follows the pointer anywhere
//! (a detent at 0 dB). Each is the core's [`DeskAction`], so the double
//! click to 0 dB and the taper are the core's; the window lets go of a cap
//! held on any release (whatever became of this widget meanwhile), and
//! turns the wheel over the strip into steps.

use iced::alignment::Vertical;
use iced::mouse::{self, Cursor, Interaction};
use iced::widget::canvas::{self, Cache, Event, Frame, Geometry, Path, Program, Stroke, Text};
use iced::widget::text::Alignment;
use iced::{Color, Element, Fill, Point, Rectangle, Renderer, Size, Theme, Widget as _};

use crate::core::desk::meter::{Meter, ZONES};
use crate::core::desk::{taper, Channel, DeskAction};
use crate::gui::style::{self, ACCENT, DIM, FAIL, GREEN, TEXT};
use crate::gui::window::Message;

/// A meter as shown: its level and its peak held (dBFS), its clip lamp.
#[derive(Clone, Copy, Debug)]
pub struct MeterLook {
    pub db: f32,
    pub peak_db: f32,
    pub clipped: bool,
}

/// The ruler the meters are read on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scale {
    /// The fader's taper, up to +6 dB at the top (the overs).
    Taper,
    /// −60 to 0 dBFS, linear in dB.
    Dbfs60,
}

/// A strip's tall section.
pub struct Fader {
    /// Whose.
    pub channel: Channel,
    /// The cap's position (0 to 1); none: meters only.
    pub cap: Option<f32>,
    pub selected: bool,
    pub meters: [MeterLook; 2],
    pub scale: Scale,
    /// The scale's numbers beside it.
    pub labels: bool,
    /// The master's wide bars, with clip lamps and their names.
    pub wide: bool,
}

/// The section as a widget.
pub fn view(f: Fader) -> Element<'static, Message> {
    canvas::Canvas::new(f).width(Fill).height(Fill).boxed()
}

/// The widget's own state: the static layer, a held cap.
#[derive(Default)]
pub struct Held {
    cache: Cache,
    /// How far the cap's middle is below the pointer, in pixels, while held.
    drag: Option<f32>,
}

/// Half the cap's height; the travel ends this far in.
const CAP_H: f32 = 22.0;
const CAP_W: f32 = 34.0;
const END: f32 = CAP_H / 2.0 + 4.0;
/// Within this many pixels of 0 dB a held cap is at 0 dB exactly.
const DETENT_PX: f32 = 5.0;
const SCALE_W: f32 = 24.0;
const METER_W: f32 = 6.0;

const METER_BG: Color = Color::from_rgb(0.07, 0.07, 0.085);

/// Where things are across and down.
struct Geo {
    top: f32,
    bottom: f32,
    /// The scale's right edge.
    scale_x: f32,
    /// The slot's middle.
    slot_x: f32,
    /// The meters' left edges and width.
    meters: [f32; 2],
    meter_w: f32,
}

impl Fader {
    fn geo(&self, size: Size) -> Geo {
        let (top, bottom) = if self.wide {
            (END + 14.0, size.height - END - 14.0)
        } else {
            (END, size.height - END)
        };
        let scale_w = if self.labels { SCALE_W } else { 0.0 };
        let (meter_w, apart) = if self.wide {
            (
                ((size.width - scale_w - 40.0) / 2.0).clamp(10.0, 26.0),
                14.0,
            )
        } else {
            (METER_W, 3.0)
        };
        let slot_w = if self.cap.is_some() { CAP_W + 8.0 } else { 6.0 };
        let group = scale_w + slot_w + meter_w * 2.0 + apart;
        let x0 = ((size.width - group) / 2.0).max(0.0);
        let scale_x = x0 + scale_w;
        let slot_x = scale_x + slot_w / 2.0;
        let m0 = scale_x + slot_w;
        Geo {
            top,
            bottom,
            scale_x,
            slot_x,
            meters: [m0, m0 + meter_w + apart],
            meter_w,
        }
    }

    /// The y of a share of the travel (0: the bottom, 1: the top).
    fn y(g: &Geo, share: f32) -> f32 {
        g.bottom - (g.bottom - g.top) * share.clamp(0.0, 1.0)
    }

    /// The share of the meters' height lit by `db` (dBFS).
    fn share(&self, db: f32) -> f32 {
        if db == f32::NEG_INFINITY {
            return 0.0;
        }
        match self.scale {
            Scale::Taper => taper::position_of(f64::from(db)) as f32,
            Scale::Dbfs60 => Meter::share(db),
        }
    }

    /// The marks: where, and what they say.
    fn marks(&self) -> Vec<(f32, String, bool)> {
        match self.scale {
            Scale::Taper => {
                let mut m: Vec<(f32, String, bool)> = taper::MARKS
                    .iter()
                    .map(|&(p, db)| (p as f32, taper::number(db, 0, true), db == 0.0))
                    .collect();
                m.push((0.0, "−∞".into(), false));
                m
            }
            Scale::Dbfs60 => [0.0, -6.0, -12.0, -18.0, -30.0, -40.0, -60.0]
                .iter()
                .map(|&db| {
                    (
                        Meter::share(db),
                        taper::number(f64::from(db), 0, true),
                        db == 0.0,
                    )
                })
                .collect(),
        }
    }

    fn position_at(&self, g: &Geo, y: f32) -> f64 {
        let unity = Self::y(g, taper::UNITY as f32);
        if (y - unity).abs() < DETENT_PX {
            return taper::UNITY;
        }
        f64::from(((g.bottom - y) / (g.bottom - g.top)).clamp(0.0, 1.0))
    }

    /// The colour of a meter at `db` dBFS.
    fn zone(db: f32) -> Color {
        if db < ZONES.0 {
            GREEN
        } else if db < ZONES.1 {
            ACCENT
        } else {
            FAIL
        }
    }

    fn draw_static(&self, frame: &mut Frame, g: &Geo) {
        let size = frame.size();
        // The ticks and the numbers.
        for (share, label, unity) in self.marks() {
            let y = Self::y(g, share);
            let color = if unity { TEXT } else { DIM };
            let from = if self.labels {
                g.scale_x + 1.0
            } else {
                g.slot_x - CAP_W / 2.0
            };
            let to = if self.cap.is_some() {
                g.slot_x + CAP_W / 2.0
            } else {
                g.slot_x + 3.0
            };
            frame.fill_rectangle(
                Point::new(from, y - 0.5),
                Size::new(to - from, 1.0),
                style::with_alpha(color, if unity { 0.7 } else { 0.35 }),
            );
            if self.labels {
                frame.fill_text(Text {
                    content: label,
                    position: Point::new(g.scale_x - 3.0, y),
                    color,
                    size: 10.5.into(),
                    font: style::digits(),
                    align_x: Alignment::Right,
                    align_y: Vertical::Center,
                    ..Text::default()
                });
            }
        }
        // The slot: a groove in the panel.
        if self.cap.is_some() {
            let slot = Path::rounded_rectangle(
                Point::new(g.slot_x - 3.0, g.top - 4.0),
                Size::new(6.0, g.bottom - g.top + 8.0),
                3.0.into(),
            );
            frame.fill(&slot, Color::from_rgb8(6, 6, 8));
            frame.stroke(
                &slot,
                Stroke::default()
                    .with_width(1.0)
                    .with_color(Color::from_rgba8(255, 255, 255, 0.06)),
            );
        }
        // The meters' faces, their zones unlit.
        let (a, b) = (self.share(ZONES.0), self.share(ZONES.1));
        for &x in &g.meters {
            let face = Path::rounded_rectangle(
                Point::new(x - 1.0, g.top - 1.0),
                Size::new(g.meter_w + 2.0, g.bottom - g.top + 2.0),
                2.0.into(),
            );
            frame.fill(&face, METER_BG);
            for (lo, hi, color) in [(0.0, a, GREEN), (a, b, ACCENT), (b, 1.0, FAIL)] {
                let (y0, y1) = (Self::y(g, hi), Self::y(g, lo));
                frame.fill_rectangle(
                    Point::new(x, y0),
                    Size::new(g.meter_w, y1 - y0),
                    style::with_alpha(color, 0.09),
                );
            }
            // Segments, as LEDs: a dark line every 3 pixels.
            let mut y = g.top + 2.0;
            while y < g.bottom {
                frame.fill_rectangle(
                    Point::new(x, y),
                    Size::new(g.meter_w, 1.0),
                    Color::from_rgba(0.0, 0.0, 0.0, 0.35),
                );
                y += 3.0;
            }
        }
        if self.wide {
            for (x, name) in g.meters.iter().zip(["L", "R"]) {
                frame.fill_text(Text {
                    content: name.into(),
                    position: Point::new(x + g.meter_w / 2.0, g.bottom + 12.0),
                    color: DIM,
                    size: 11.0.into(),
                    font: style::ui(),
                    align_x: Alignment::Center,
                    align_y: Vertical::Center,
                    ..Text::default()
                });
            }
        }
        let _ = size;
    }

    fn draw_moving(&self, frame: &mut Frame, g: &Geo, hover: bool) {
        // The lit bars, by zone, brighter at their tops; the ticks above.
        let (a, b) = (self.share(ZONES.0), self.share(ZONES.1));
        for (m, &x) in self.meters.iter().zip(&g.meters) {
            let lit = self.share(m.db);
            for (lo, hi, color) in [(0.0, a, GREEN), (a, b, ACCENT), (b, 1.0, FAIL)] {
                let top = lit.min(hi);
                if top <= lo {
                    continue;
                }
                let (y0, y1) = (Self::y(g, top), Self::y(g, lo));
                frame.fill_rectangle(Point::new(x, y0), Size::new(g.meter_w, y1 - y0), color);
            }
            if lit > 0.0 {
                // A glow at the bar's top.
                let y = Self::y(g, lit);
                frame.fill_rectangle(
                    Point::new(x - 1.0, y - 1.0),
                    Size::new(g.meter_w + 2.0, 2.0),
                    style::with_alpha(Color::WHITE, 0.35),
                );
            }
            if m.peak_db > m.db && m.peak_db != f32::NEG_INFINITY {
                let y = Self::y(g, self.share(m.peak_db));
                frame.fill_rectangle(
                    Point::new(x, y - 1.0),
                    Size::new(g.meter_w, 2.0),
                    Self::zone(m.peak_db),
                );
            }
            if self.wide {
                // The clip lamp over it.
                let c = Point::new(x + g.meter_w / 2.0, g.top - 12.0);
                if m.clipped {
                    frame.fill(&Path::circle(c, 8.0), style::with_alpha(FAIL, 0.25));
                }
                frame.fill(
                    &Path::circle(c, 4.0),
                    if m.clipped {
                        FAIL
                    } else {
                        Color::from_rgb8(46, 46, 54)
                    },
                );
            }
        }
        // The cap, with its shadow and its line.
        if let Some(p) = self.cap {
            let y = Self::y(g, p);
            let x = g.slot_x - CAP_W / 2.0;
            let shadow = Path::rounded_rectangle(
                Point::new(x + 1.0, y - CAP_H / 2.0 + 4.0),
                Size::new(CAP_W, CAP_H),
                4.0.into(),
            );
            frame.fill(&shadow, Color::from_rgba(0.0, 0.0, 0.0, 0.45));
            let body = Path::rounded_rectangle(
                Point::new(x, y - CAP_H / 2.0),
                Size::new(CAP_W, CAP_H),
                4.0.into(),
            );
            let lift = if hover { 0.08 } else { 0.0 };
            let gradient = canvas::gradient::Linear::new(
                Point::new(0.0, y - CAP_H / 2.0),
                Point::new(0.0, y + CAP_H / 2.0),
            )
            .add_stop(
                0.0,
                style::mix(Color::from_rgb8(214, 214, 222), Color::WHITE, lift),
            )
            .add_stop(
                0.45,
                style::mix(Color::from_rgb8(150, 150, 160), Color::WHITE, lift),
            )
            .add_stop(
                0.55,
                style::mix(Color::from_rgb8(120, 120, 130), Color::WHITE, lift),
            )
            .add_stop(
                1.0,
                style::mix(Color::from_rgb8(78, 78, 88), Color::WHITE, lift),
            );
            frame.fill(&body, gradient);
            frame.stroke(
                &body,
                Stroke::default()
                    .with_width(1.0)
                    .with_color(Color::from_rgb8(20, 20, 24)),
            );
            // The grip's ridges, and the line it is read by.
            for dy in [-6.0, -3.0, 3.0, 6.0] {
                frame.fill_rectangle(
                    Point::new(x + 5.0, y + dy - 0.5),
                    Size::new(CAP_W - 10.0, 1.0),
                    Color::from_rgba(0.0, 0.0, 0.0, 0.25),
                );
            }
            frame.fill_rectangle(
                Point::new(x + 2.0, y - 1.0),
                Size::new(CAP_W - 4.0, 2.0),
                if self.selected {
                    ACCENT
                } else {
                    Color::from_rgb8(16, 16, 20)
                },
            );
        }
    }

    fn cap_rect(&self, g: &Geo) -> Option<Rectangle> {
        let p = self.cap?;
        let y = Self::y(g, p);
        Some(Rectangle::new(
            Point::new(g.slot_x - CAP_W / 2.0, y - CAP_H / 2.0),
            Size::new(CAP_W, CAP_H),
        ))
    }
}

impl Program<Message> for Fader {
    type State = Held;

    fn update(
        &self,
        held: &mut Held,
        event: &Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> Option<canvas::Action<Message>> {
        let g = self.geo(bounds.size());
        let Event::Mouse(event) = event else {
            return None;
        };
        let strip = match self.channel {
            Channel::Strip(kind) => Some(kind),
            _ => None,
        };
        // Where the pointer is, in the widget, even outside it (a held cap
        // follows it anywhere).
        let at = cursor
            .position()
            .map(|p| Point::new(p.x - bounds.x, p.y - bounds.y));
        match event {
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let at = cursor.position_in(bounds)?;
                let kind = strip?;
                let offset = match self.cap_rect(&g) {
                    Some(r) if r.contains(at) => r.center_y() - at.y,
                    _ => 0.0,
                };
                held.drag = Some(offset);
                let position = self.position_at(&g, at.y + offset);
                Some(
                    canvas::Action::publish(Message::Desk(DeskAction::Grab {
                        strip: kind,
                        position,
                    }))
                    .and_capture(),
                )
            }
            mouse::Event::CursorMoved { .. } => {
                let offset = held.drag?;
                let at = at?;
                Some(
                    canvas::Action::publish(Message::Desk(DeskAction::Slide(
                        self.position_at(&g, at.y + offset),
                    )))
                    .and_capture(),
                )
            }
            // The window sends the release (it hears every one); here the
            // cap is no longer held.
            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                held.drag = None;
                None
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        held: &Held,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> Vec<Geometry> {
        let g = self.geo(bounds.size());
        let still = held
            .cache
            .draw(renderer, bounds.size(), |frame| self.draw_static(frame, &g));
        let mut frame = Frame::new(renderer, bounds.size());
        let hover = held.drag.is_some()
            || cursor
                .position_in(bounds)
                .zip(self.cap_rect(&g))
                .is_some_and(|(p, r)| r.contains(p));
        self.draw_moving(&mut frame, &g, hover);
        vec![still, frame.into_geometry()]
    }

    fn mouse_interaction(&self, held: &Held, bounds: Rectangle, cursor: Cursor) -> Interaction {
        if held.drag.is_some() {
            return Interaction::Grabbing;
        }
        let g = self.geo(bounds.size());
        match (cursor.position_in(bounds), self.cap_rect(&g)) {
            (Some(p), Some(r)) if r.contains(p) => Interaction::Grab,
            (Some(_), Some(_)) => Interaction::Pointer,
            _ => Interaction::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::desk::StripKind;

    fn fader() -> Fader {
        let silent = MeterLook {
            db: f32::NEG_INFINITY,
            peak_db: f32::NEG_INFINITY,
            clipped: false,
        };
        Fader {
            channel: Channel::Strip(StripKind::Player),
            cap: Some(taper::UNITY as f32),
            selected: false,
            meters: [silent; 2],
            scale: Scale::Taper,
            labels: true,
            wide: false,
        }
    }

    /// The cap goes where the pointer puts it, to the pixel, the ends
    /// beyond the travel; within [`DETENT_PX`] of 0 dB it is at 0 dB
    /// exactly, the detent unity gain depends on landing in.
    #[test]
    fn the_cap_is_where_the_pointer_puts_it_with_a_detent_at_unity() {
        let f = fader();
        let g = f.geo(Size::new(120.0, 400.0));
        assert_eq!(f.position_at(&g, g.top), 1.0);
        assert_eq!(f.position_at(&g, g.bottom), 0.0);
        assert_eq!(f.position_at(&g, g.bottom + 300.0), 0.0);
        assert_eq!(f.position_at(&g, -50.0), 1.0);
        for p in [0.0, 0.25, 0.5, 1.0] {
            let at = f.position_at(&g, Fader::y(&g, p));
            assert!((at - f64::from(p)).abs() < 1e-5, "{p}: {at}");
        }
        let unity = Fader::y(&g, taper::UNITY as f32);
        for dy in [-4.9, -2.0, 0.0, 3.0, 4.9] {
            let at = f.position_at(&g, unity + dy);
            assert_eq!(taper::db_at(at), 0.0, "{dy}");
        }
        assert!(taper::db_at(f.position_at(&g, unity + 5.1)) < 0.0);
        assert!(taper::db_at(f.position_at(&g, unity - 5.1)) > 0.0);
    }
}
