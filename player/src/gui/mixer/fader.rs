//! A strip's tall section as a canvas: the scale, the fader's slot and its
//! cap, and the meters with their peak-hold ticks and clip lamps, in the
//! look's tones ([`crate::gui::look`]): the meters in the status colours
//! (the accent their middle zone), the selected fader's slot lit in the
//! accent up to its cap. Edges fall on whole points, so it stays crisp.
//! What stands still (the scale, the slot, the meters' unlit faces) is
//! drawn once into a cache, again only when the size changes; the cap and
//! the lit bars every frame, a few rectangles.
//!
//! The mouse: a press on the cap takes it where it is, elsewhere on the
//! slot brings it to the pointer; held, it follows the pointer anywhere
//! (a detent at 0 dB). Each is the core's [`DeskAction`], so the double
//! click to 0 dB and the taper are the core's; the window lets go of a cap
//! held on any release (whatever became of this widget meanwhile), and
//! turns the wheel over the strip into steps.

use iced::alignment::Vertical;
use iced::mouse::{self, Cursor, Interaction};
use iced::widget::canvas::{self, Cache, Event, Frame, Geometry, Path, Program, Text};
use iced::widget::text::Alignment;
use iced::{Color, Element, Fill, Point, Rectangle, Renderer, Size, Theme, Widget as _};

use crate::core::desk::meter::{Meter, ZONES};
use crate::core::desk::{taper, Channel, DeskAction};
use crate::core::Action;
use crate::gui::look::{self, size};
use crate::gui::style::mix;
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
    /// How far the cap's middle is below the pointer, in points, while held.
    drag: Option<f32>,
}

/// The cap's size, in points; the travel ends a little past its half.
const CAP_H: f32 = 20.0;
const CAP_W: f32 = 30.0;
const END: f32 = CAP_H / 2.0 + 4.0;
/// Within this many points of 0 dB a held cap is at 0 dB exactly.
const DETENT: f32 = 5.0;
/// The scale's numbers' column, and a meter's width, in points.
const SCALE_W: f32 = 26.0;
const METER_W: f32 = 5.0;
/// The slot's width.
const SLOT: f32 = 4.0;
/// A wide meter's room for its clip lamp above it and its name below.
const LAMP_ROOM: f32 = 16.0;

/// The meters' and the slot's ground: darker than the strip.
const WELL: Color = Color::from_rgb8(0x0b, 0x0b, 0x0e);
/// A lamp not lit.
const UNLIT: Color = look::EDGE;

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
            (END + LAMP_ROOM, size.height - END - LAMP_ROOM)
        } else {
            (END, size.height - END)
        };
        let scale_w = if self.labels { SCALE_W } else { 0.0 };
        let (meter_w, apart) = if self.wide {
            (
                ((size.width - scale_w - 40.0) / 2.0)
                    .clamp(10.0, 24.0)
                    .floor(),
                12.0,
            )
        } else {
            (METER_W, 2.0)
        };
        let slot_w = if self.cap.is_some() { CAP_W + 6.0 } else { 8.0 };
        let group = scale_w + slot_w + meter_w * 2.0 + apart;
        let x0 = ((size.width - group) / 2.0).max(0.0).floor();
        let scale_x = x0 + scale_w;
        let slot_x = scale_x + slot_w / 2.0;
        let m0 = scale_x + slot_w;
        Geo {
            top: top.round(),
            bottom: bottom.round(),
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
        if (y - unity).abs() < DETENT {
            return taper::UNITY;
        }
        f64::from(((g.bottom - y) / (g.bottom - g.top)).clamp(0.0, 1.0))
    }

    /// The colour of a meter at `db` dBFS: good, the accent near the top,
    /// fail at the top.
    fn zone(db: f32) -> Color {
        if db < ZONES.0 {
            look::GOOD
        } else if db < ZONES.1 {
            look::ACCENT
        } else {
            look::FAIL
        }
    }

    /// The meters' zones: from, to (shares of the height), and colour.
    fn zones(&self) -> [(f32, f32, Color); 3] {
        let (a, b) = (self.share(ZONES.0), self.share(ZONES.1));
        [
            (0.0, a, look::GOOD),
            (a, b, look::ACCENT),
            (b, 1.0, look::FAIL),
        ]
    }

    /// The span from share `lo` to `hi` of the meter at `x`, on whole
    /// points.
    fn span(frame: &mut Frame, g: &Geo, x: f32, (lo, hi): (f32, f32), color: Color) {
        let (y0, y1) = (Self::y(g, hi).round(), Self::y(g, lo).round());
        if y1 > y0 {
            frame.fill_rectangle(Point::new(x, y0), Size::new(g.meter_w, y1 - y0), color);
        }
    }

    fn draw_static(&self, frame: &mut Frame, g: &Geo) {
        // The ticks, and the numbers where they do not crowd each other
        // (0 dB's first).
        let mut marks = self.marks();
        marks.sort_by_key(|&(_, _, unity)| !unity);
        let mut numbered: Vec<f32> = Vec::new();
        for (share, label, unity) in marks {
            let y = Self::y(g, share).round();
            let (from, to) = if self.cap.is_some() {
                (g.slot_x - CAP_W / 2.0 + 3.0, g.slot_x + CAP_W / 2.0 - 3.0)
            } else {
                (g.slot_x - 3.0, g.slot_x + 3.0)
            };
            frame.fill_rectangle(
                Point::new(from.round(), y - 0.5),
                Size::new((to - from).round(), 1.0),
                if unity { look::TEXT_FAINT } else { look::EDGE },
            );
            if self.labels && numbered.iter().all(|&at| (at - y).abs() > size::CAPTION) {
                numbered.push(y);
                frame.fill_text(Text {
                    content: label,
                    position: Point::new(g.scale_x - 2.0, y),
                    color: if unity {
                        look::TEXT_DIM
                    } else {
                        look::TEXT_FAINT
                    },
                    size: size::CAPTION.into(),
                    font: look::font::numbers(),
                    align_x: Alignment::Right,
                    align_y: Vertical::Center,
                    ..Text::default()
                });
            }
        }
        // The slot: a thin groove, darker than the strip.
        if self.cap.is_some() {
            let slot = Path::rounded_rectangle(
                Point::new((g.slot_x - SLOT / 2.0).round(), g.top),
                Size::new(SLOT, g.bottom - g.top),
                (SLOT / 2.0).into(),
            );
            frame.fill(&slot, WELL);
        }
        // The meters' wells, their zones faint.
        for &x in &g.meters {
            let well = Path::rounded_rectangle(
                Point::new(x - 1.0, g.top - 1.0),
                Size::new(g.meter_w + 2.0, g.bottom - g.top + 2.0),
                2.0.into(),
            );
            frame.fill(&well, WELL);
            for (lo, hi, color) in self.zones() {
                Self::span(frame, g, x, (lo, hi), look::tint(color, WELL, 0.12));
            }
        }
        if self.wide {
            for (x, name) in g.meters.iter().zip(["L", "R"]) {
                frame.fill_text(Text {
                    content: name.into(),
                    position: Point::new(x + g.meter_w / 2.0, g.bottom + LAMP_ROOM / 2.0 + 2.0),
                    color: look::TEXT_FAINT,
                    size: size::CAPTION.into(),
                    font: look::font::label(),
                    align_x: Alignment::Center,
                    align_y: Vertical::Center,
                    ..Text::default()
                });
            }
        }
    }

    fn draw_moving(&self, frame: &mut Frame, g: &Geo, hover: bool) {
        // The lit bars, by zone; the peaks held above them.
        for (m, &x) in self.meters.iter().zip(&g.meters) {
            let lit = self.share(m.db);
            for (lo, hi, color) in self.zones() {
                let top = lit.min(hi);
                if top > lo {
                    Self::span(frame, g, x, (lo, top), color);
                }
            }
            if m.peak_db > m.db && m.peak_db != f32::NEG_INFINITY {
                let y = Self::y(g, self.share(m.peak_db)).round();
                frame.fill_rectangle(
                    Point::new(x, y - 1.0),
                    Size::new(g.meter_w, 2.0),
                    Self::zone(m.peak_db),
                );
            }
            if self.wide {
                // The clip lamp over it.
                let c = Point::new(x + g.meter_w / 2.0, g.top - LAMP_ROOM / 2.0 - 2.0);
                if m.clipped {
                    frame.fill(&Path::circle(c, 7.0), look::alpha(look::FAIL, 0.25));
                }
                frame.fill(
                    &Path::circle(c, 4.0),
                    if m.clipped { look::FAIL } else { UNLIT },
                );
            }
        }
        // The cap, the slot under it lit up to it while selected.
        if let Some(p) = self.cap {
            let y = Self::y(g, p).round();
            let slot_x = (g.slot_x - SLOT / 2.0).round();
            if self.selected && y < g.bottom {
                frame.fill(
                    &Path::rounded_rectangle(
                        Point::new(slot_x, y),
                        Size::new(SLOT, g.bottom - y),
                        (SLOT / 2.0).into(),
                    ),
                    look::ACCENT,
                );
            }
            let x = (g.slot_x - CAP_W / 2.0).round();
            let top = y - CAP_H / 2.0;
            frame.fill(
                &Path::rounded_rectangle(
                    Point::new(x, top + 2.0),
                    Size::new(CAP_W, CAP_H),
                    look::radius::CONTROL.into(),
                ),
                Color::from_rgba(0.0, 0.0, 0.0, 0.4),
            );
            let body = mix(look::SELECTED, look::TEXT, if hover { 0.4 } else { 0.28 });
            frame.fill(
                &Path::rounded_rectangle(
                    Point::new(x, top),
                    Size::new(CAP_W, CAP_H),
                    look::radius::CONTROL.into(),
                ),
                body,
            );
            // A lighter top edge, and the line it is read by.
            frame.fill_rectangle(
                Point::new(x + 4.0, top),
                Size::new(CAP_W - 8.0, 1.0),
                mix(body, look::TEXT, 0.35),
            );
            frame.fill_rectangle(
                Point::new(x + 4.0, y - 1.0),
                Size::new(CAP_W - 8.0, 2.0),
                if self.selected {
                    look::ACCENT
                } else {
                    look::TEXT
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
                    canvas::Action::publish(Message::Act(Action::Desk(DeskAction::Grab {
                        strip: kind,
                        position,
                    })))
                    .and_capture(),
                )
            }
            mouse::Event::CursorMoved { .. } => {
                let offset = held.drag?;
                let at = at?;
                Some(
                    canvas::Action::publish(Message::Act(Action::Desk(DeskAction::Slide(
                        self.position_at(&g, at.y + offset),
                    ))))
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

    /// The cap goes where the pointer puts it, to the point, the ends
    /// beyond the travel; within [`DETENT`] of 0 dB it is at 0 dB exactly,
    /// the detent unity gain depends on landing in.
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
