//! The main screen's measures, in the window's points: the tabs' header
//! along the top, the side pane under it with the right column beside it
//! (or the track's summary under it, in a narrow window), the Karaoke
//! tab's voice lane under the pane, and the bottom bar along the foot
//! (the spectrum at its top on the browser's and the queue's sides). The
//! window lays the screen out in iced's rows and columns from these
//! ([`super::screens`]): the panes fill what the fixed parts leave
//! ([`main`] says where that is), and what the window's size does to the
//! keys and the pointer comes from the same numbers ([`fit`]).
//!
//! The thresholds are the old cell layout's at its cells' size at a
//! scale of 1 (8 × 19 points): the right column goes below 576 points (72
//! columns), the spectrum shrinks with the height (30, 24 and 18 rows).

use iced::{Point, Rectangle, Size};

use super::look::space;
use crate::core::{Fit, Screen, Side, State};

/// The tabs' header along the window's top.
pub const HEADER: f32 = 36.0;
/// The tabs' segmented control in it.
pub const TABS: f32 = 28.0;
/// The bottom bar's band under its spectrum: the seek bar's line and the
/// buttons.
pub const BAND: f32 = 60.0;
/// The 1 px edge over the bottom bar.
pub const EDGE: f32 = 1.0;
/// Round the panes, at the window's sides and over the bottom bar.
pub const MARGIN: f32 = space::S;
/// Between the side pane and the right column.
pub const GAP: f32 = space::L;
/// The track's summary under the list in a narrow window.
pub const SUMMARY: f32 = 40.0;
/// The least room under the header, summary and all, for the summary to
/// go under the side pane (the old 10 rows).
const SUMMARY_ROOM: f32 = 190.0;
/// A line of the old cells' height: what a wheel's line scrolls.
pub const LINE: f32 = 19.0;
/// The right column's text under the cover: the cover gives way to it.
const INFO: f32 = 152.0;
/// Between the cover and the text under it.
pub const UNDER_COVER: f32 = space::L;
/// The least the right column's cover is shown at.
const LEAST_COVER: f32 = 76.0;

/// The spectrum's height over the bottom bar in a window `height` points
/// high (0: none): 190 points from 570 high, 133 from 456, 76 from 342.
pub fn spectrum(height: f32) -> f32 {
    match height {
        h if h >= 570.0 => 190.0,
        h if h >= 456.0 => 133.0,
        h if h >= 342.0 => 76.0,
        _ => 0.0,
    }
}

/// The right column's width in a window `width` points wide (0: none):
/// the same whatever plays, so it stays put from track to track.
pub fn right_width(width: f32) -> f32 {
    match width {
        w if w >= 960.0 => (w * 0.34).min(480.0).floor(),
        w if w >= 768.0 => 272.0,
        w if w >= 576.0 => 224.0,
        _ => 0.0,
    }
}

/// The main screen laid out in a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Main {
    /// The tabs' header.
    pub header: Rectangle,
    /// The side pane: the browser, the queue or a tab of its own.
    pub pane: Rectangle,
    /// The right column beside it (the browser's and the queue's sides,
    /// where the window is wide enough).
    pub right: Option<Rectangle>,
    /// The right column's cover's place, at its top (where it has room).
    pub cover: Option<Rectangle>,
    /// The track's summary under the pane, where there is no right column.
    pub summary: Option<Rectangle>,
    /// The Karaoke tab's voice lane, under the pane (where the other sides
    /// have the spectrum).
    pub voice: Option<Rectangle>,
    /// The bottom bar, its spectrum at its top.
    pub bar: Rectangle,
    /// The spectrum's height at the bar's top (0: none).
    pub spectrum: f32,
}

/// The main screen with `side` showing in a window of `size`.
pub fn main(size: Size, side: Side) -> Main {
    let (w, h) = (size.width, size.height);
    let wide = side.wide();
    let strip = spectrum(h);
    let spectrum = if wide { 0.0 } else { strip };
    let voice_h = if side == Side::Karaoke { strip } else { 0.0 };
    let bar_h = spectrum + EDGE + BAND;
    let bar = Rectangle::new(Point::new(0.0, h - bar_h), Size::new(w, bar_h));
    let rw = if wide { 0.0 } else { right_width(w) };
    let inner = (w - 2.0 * MARGIN).max(0.0);
    let pane_w = if rw > 0.0 {
        (inner - rw - GAP).max(0.0)
    } else {
        inner
    };
    let voice_room = if voice_h > 0.0 { voice_h + MARGIN } else { 0.0 };
    let top = (h - HEADER - MARGIN - bar_h - voice_room).max(0.0);
    let summary_room = SUMMARY + MARGIN;
    let summary = !wide && rw == 0.0 && top >= SUMMARY_ROOM;
    let pane_h = if summary { top - summary_room } else { top };
    let pane = Rectangle::new(Point::new(MARGIN, HEADER), Size::new(pane_w, pane_h));
    let below = HEADER + pane_h + MARGIN;
    let right = (rw > 0.0).then(|| {
        Rectangle::new(
            Point::new(MARGIN + pane_w + GAP, HEADER),
            Size::new(rw, pane_h),
        )
    });
    let cover = right.and_then(|r| {
        let side = r.width.min(r.height - INFO - UNDER_COVER);
        (side >= LEAST_COVER).then(|| Rectangle::new(r.position(), Size::new(r.width, side)))
    });
    Main {
        header: Rectangle::new(Point::ORIGIN, Size::new(w, HEADER)),
        pane,
        right,
        cover,
        summary: summary
            .then(|| Rectangle::new(Point::new(MARGIN, below), Size::new(inner, SUMMARY))),
        voice: (voice_h > 0.0)
            .then(|| Rectangle::new(Point::new(MARGIN, below), Size::new(inner, voice_h))),
        bar,
        spectrum,
    }
}

/// The height of the page over the bottom bar on a screen the window lays
/// out whole (the settings, the keys) in a window `height` points high:
/// the bar there has no spectrum.
pub fn page(height: f32) -> f32 {
    (height - EDGE - BAND).max(0.0)
}

/// What a window of `size` does to the keys and the pointer, for the
/// core ([`crate::core::Action::Fit`]), whichever screen and side show
/// (the list keeps its page behind the other tabs, the keys go round the
/// desk's strips from any tab): the list's page and the help's, the
/// desk's width, the Transition tab's lanes in letters `letter` points
/// wide each (the lanes' words take a letter each), Karaoke mode's
/// controls' reach, and whether a spectrum shows.
pub fn fit(size: Size, state: &State, letter: f32) -> Fit {
    let list = main(size, Side::Browser);
    let wide = main(size, Side::Mixer);
    let shown = main(size, state.side);
    Fit {
        list_rows: super::screens::list::rows_in(list.pane.height),
        help_rows: crate::view::help::rows(page(size.height), state),
        desk_width: wide.pane.width,
        lane_letters: crate::view::transition::lane_letters(wide.pane.width, letter),
        controls_reach: super::screens::karaoke::mode::reach(size.height),
        spectrum: state.screen == Screen::Main && shown.spectrum > 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old 120 columns (960 points) and 80 (640) at 36 and 24 rows:
    /// the right column and the spectrum as the cells had them; the parts
    /// fill the window without overlapping.
    #[test]
    fn the_parts_fill_the_window_at_the_old_widths() {
        let at = |w: f32, h: f32| main(Size::new(w, h), Side::Browser);
        let wide = at(960.0, 684.0);
        assert_eq!(wide.right.map(|r| r.width), Some(326.0));
        assert_eq!(wide.spectrum, 190.0);
        assert!(wide.cover.is_some() && wide.summary.is_none());
        let narrow = at(640.0, 456.0);
        assert_eq!(narrow.right.map(|r| r.width), Some(224.0));
        assert_eq!(narrow.spectrum, 133.0);
        let phone = at(512.0, 418.0);
        assert!(phone.right.is_none() && phone.summary.is_some());
        for m in [wide, narrow, phone] {
            let pane = m.pane;
            assert_eq!(pane.y, HEADER);
            let foot = m.summary.map_or(pane.y + pane.height, |s| s.y + s.height);
            assert_eq!(foot + MARGIN, m.bar.y, "{m:?}");
            if let Some(r) = m.right {
                assert_eq!(pane.x + pane.width + GAP, r.x);
                assert_eq!(r.x + r.width + MARGIN, m.bar.width);
            }
        }
    }

    /// A tab of its own has the whole width and no spectrum; the Karaoke
    /// tab has its voice's lane where the spectrum goes, over the bar.
    #[test]
    fn a_wide_tab_takes_the_width_and_the_karaoke_tab_its_lane() {
        let size = Size::new(1280.0, 800.0);
        let mixer = main(size, Side::Mixer);
        assert!(mixer.right.is_none() && mixer.voice.is_none());
        assert_eq!(mixer.pane.width, 1280.0 - 2.0 * MARGIN);
        assert_eq!(mixer.spectrum, 0.0);
        let karaoke = main(size, Side::Karaoke);
        let voice = karaoke.voice.unwrap();
        assert_eq!(voice.height, 190.0);
        assert_eq!(voice.y + voice.height + MARGIN, karaoke.bar.y);
        assert_eq!(karaoke.pane.y + karaoke.pane.height + MARGIN, voice.y);
    }

    /// The desk's widths of the rules (M6): every strip on a window 960
    /// wide, the old 120 columns, as on 944 points of desk.
    #[test]
    fn the_desk_is_the_window_less_its_margins() {
        assert_eq!(main(Size::new(960.0, 600.0), Side::Mixer).pane.width, 944.0);
        assert_eq!(main(Size::new(640.0, 600.0), Side::Mixer).pane.width, 624.0);
    }
}
