//! The Karaoke tab's layout in cells (its cover, title, button into the
//! mode, sliders, switches and the sync's panel) and Karaoke mode's (the
//! whole screen, and its controls' bar while they show).

use ratatui::layout::Rect;

use super::square_in;
use crate::core::karaoke;
use crate::core::keymap::Op;
use crate::core::sliders::Slider;
use crate::core::State;
use crate::view::cells::width;

/// A Karaoke slider's cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliderLayout {
    /// Which.
    pub slider: Slider,
    /// Its row: the label, the track and the value.
    pub row: Rect,
    /// The track its knob moves along (a click or drag sets it).
    pub track: Rect,
    /// The value, right of the track.
    pub value: Rect,
}

/// The Karaoke tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeTabLayout {
    /// The playing track's cover.
    pub cover: Rect,
    /// The title and artist, right of it.
    pub title: Rect,
    /// The button into Karaoke mode.
    pub enter: Rect,
    /// The sliders.
    pub sliders: Vec<SliderLayout>,
    /// The game on or off: its row, a click turns it.
    pub game: Rect,
    /// The notes shown or hidden: its row.
    pub show_notes: Rect,
    /// The singing kept or not: its row.
    pub keep: Rect,
    /// The sync's row: its button (a click starts one), then the delay in
    /// force.
    pub sync: Rect,
    /// The sync's panel over the tab, while one shows.
    pub panel: Option<Rect>,
    /// Its buttons (keep, again, cancel, as the sync stands), and what
    /// each does.
    pub panel_buttons: Vec<(Rect, Op)>,
    /// What karaoke will do, below.
    pub notes: Rect,
    /// The voice's lane with the words on it, under the pane (where the
    /// other tabs have the spectrum).
    pub voice: Option<Rect>,
}

/// Karaoke mode: the whole screen is its picture, and its controls (while
/// they show) a bar over its bottom (the picture stays where it is).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeLayout {
    /// The whole screen.
    pub screen: Rect,
    /// The controls, while they show.
    pub controls: Option<KaraokeControls>,
}

/// Karaoke mode's controls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeControls {
    /// The bar they are in ([`karaoke::controls::bar`]).
    pub bar: Rect,
    /// The sliders.
    pub sliders: Vec<SliderLayout>,
    /// The time heard, the seek bar and the length, under the sliders.
    pub elapsed: Rect,
    /// The seek bar.
    pub seek: Rect,
    /// The length.
    pub total: Rect,
    /// The keys, in a line.
    pub hint: Rect,
}

/// The label of the button into Karaoke mode.
pub const KARAOKE_ENTER: &str = "▶  Enter Karaoke Mode";
/// A slider's label column, in cells.
pub const SLIDER_LABEL: u16 = 12;
/// How wide a switch on the Karaoke tab is (" on " or " off ").
pub const SWITCH: u16 = 5;
/// A slider's value column (`100 %`, `−200 ms`), in cells.
const SLIDER_VALUE: u16 = 7;
/// How wide the sync's button is (` sync `).
pub const SYNC_BUTTON: u16 = 6;
/// The sync's panel's height: over the button into the mode and every
/// row of settings.
const PANEL: u16 = 12;
/// The longest a slider's track gets, in cells.
const SLIDER_MOST: u16 = 40;

/// A slider in the row at `x`, `y`, `width` cells wide: its label, the
/// track (up to [`SLIDER_MOST`]) and the value after it.
fn slider_at(slider: Slider, x: u16, y: u16, width: u16) -> SliderLayout {
    let track = width
        .saturating_sub(SLIDER_LABEL + SLIDER_VALUE + 1)
        .min(SLIDER_MOST);
    let row = Rect::new(x, y, SLIDER_LABEL + track + 1 + SLIDER_VALUE, 1);
    SliderLayout {
        slider,
        row,
        track: Rect::new(x + SLIDER_LABEL, y, track, 1),
        value: Rect::new(x + SLIDER_LABEL + track + 1, y, SLIDER_VALUE, 1),
    }
}

/// The Karaoke tab inside the list pane's border: the cover at the left,
/// the title, the button into the mode, the sliders and the sync's row
/// right of it, notes below; while a sync shows (`session`), its panel
/// over the button and the rows.
pub(super) fn tab(inner: Rect, session: Option<&crate::core::sync::Session>) -> KaraokeTabLayout {
    // The cover as tall as the pane allows, a third of its width at most.
    let side = Rect::new(
        inner.x + 1,
        inner.y + 1,
        (inner.width / 3).min(inner.height.saturating_sub(2).saturating_mul(2)),
        inner.height.saturating_sub(2),
    );
    let cover = square_in(side);
    let cover = Rect { y: side.y, ..cover };
    let x = if cover.width > 0 {
        cover.right() + 3
    } else {
        inner.x + 2
    };
    let room = inner.right().saturating_sub(x + 1);
    let row = |k: u16| (inner.y + 1 + k).min(inner.bottom().saturating_sub(1));
    let title = Rect::new(x, row(0), room, 2.min(inner.height));
    let enter_w = (width(KARAOKE_ENTER) as u16 + 4).min(room);
    let enter = Rect::new(x, row(3), enter_w, 3.min(inner.height.saturating_sub(4)));
    // A row each from the seventh, the sync's row right after the delay's.
    let (mut at, mut sync_row, mut sliders) = (7, 0, Vec::new());
    for &s in &Slider::ALL {
        sliders.push(slider_at(s, x, row(at), room));
        at += 1;
        if s == Slider::Delay {
            (sync_row, at) = (at, at + 1);
        }
    }
    let sync = Rect::new(x, row(sync_row), (SLIDER_LABEL + SYNC_BUTTON).min(room), 1);
    // Over the button into the mode and the rows below it.
    let panel =
        session.map(|_| Rect::new(x, row(3), room, PANEL.min(inner.height.saturating_sub(4))));
    // The panel's buttons, side by side on its eighth row inside.
    let mut panel_buttons = Vec::new();
    if let (Some(p), Some(session)) = (panel, session) {
        let mut at = p.x + 2;
        for (label, op) in session.buttons() {
            let w = width(label) as u16 + 2;
            let r = Rect::new(at, p.y + 8, w, 1);
            if r.right() < p.right() && r.y < p.bottom().saturating_sub(1) {
                panel_buttons.push((r, op));
            }
            at += w + 2;
        }
    }
    // The game's settings, a switch a row.
    let game = Rect::new(x, row(11), (SLIDER_LABEL + SWITCH).min(room), 1);
    let show_notes = Rect::new(x, row(12), (SLIDER_LABEL + SWITCH).min(room), 1);
    let keep = Rect::new(x, row(13), (SLIDER_LABEL + SWITCH).min(room), 1);
    let notes_y = row(15);
    let notes = Rect::new(x, notes_y, room, inner.bottom().saturating_sub(notes_y));
    KaraokeTabLayout {
        cover,
        title,
        enter,
        sliders,
        game,
        show_notes,
        keep,
        sync,
        panel,
        panel_buttons,
        notes,
        voice: None,
    }
}

/// Karaoke mode: the whole screen, and the controls' bar over its bottom
/// while they show (the sliders, the seek bar and the keys; START is the
/// stage's pill, above the bar). The game's settings are not here but on
/// the tab: they stay as they are from a song's start to its end.
pub(super) fn mode(state: &State, area: Rect) -> KaraokeLayout {
    let controls = karaoke::controls::shown(state).then(|| {
        let bar = karaoke::controls::bar(area);
        let room = bar.width.saturating_sub(4);
        let sliders = Slider::MODE
            .iter()
            .enumerate()
            .map(|(k, &s)| slider_at(s, bar.x + 2, bar.y + 1 + k as u16, room))
            .collect();
        // The seek bar as on the other screens: the time, the bar, the
        // length.
        let seek_y = bar.y + 1 + Slider::MODE.len() as u16;
        let time_w = 8u16;
        let elapsed = Rect::new(bar.x + 1, seek_y, time_w.min(room), 1);
        let total = Rect::new(
            (bar.x + 2 + room).saturating_sub(time_w),
            seek_y,
            time_w.min(room),
            1,
        );
        let seek = Rect::new(
            elapsed.right() + 1,
            seek_y,
            total.x.saturating_sub(elapsed.right() + 2),
            1,
        );
        KaraokeControls {
            bar,
            sliders,
            elapsed,
            seek,
            total,
            hint: Rect::new(bar.x + 2, seek_y + 1, room, 1),
        }
    });
    KaraokeLayout {
        screen: area,
        controls,
    }
}
