//! The Mix tab's desk in cells: the strips side by side, the master at the
//! right, as on a desk.
//!
//! ```text
//! ╭──────────╮ ╭──────────╮ ╭──────────────╮   ╭────────────────────────╮
//! │  PLAYER  │ │ PREVIEW  │ │     MIC      │   │      MAIN · DAC        │
//! │ source…  │ │ A −3.2   │ │ name, echo…  │   │ backend, format, lamps │
//! │ +6 │  ▂▂ │ │ +6 │     │ │ 0 ┤ IN VOICE │   │  0 ┤  ███   ███        │
//! │  0▐█▌ ██ │ │  0▐█▌    │ │   ┤ ▅▅  ▃▃   │   │−18 ┤  ███   ███        │
//! │ −∞ │  ██ │ │ −∞ │     │ │   ┤ ▅▅  ▃▃   │   │−60 ┤  ███   ███        │
//! │  0.0 dB  │ │  0.0 dB  │ │ LISTEN ONLY  │   │ VOLUME ▮▮▮▮▮▮▯▯ −12 dB │
//! │ [M] [S]  │ │ [M] [S]  │ │ PITCH○ REC○  │   │                        │
//! ╰──────────╯ ╰──────────╯ ╰──────────────╯   ╰────────────────────────╯
//! ```
//!
//! Each strip's tall section takes the height left; strips grow from their
//! least widths up to their widest, the desk centred. Where it is too
//! narrow, the microphone goes first, then the preview.

use ratatui::layout::Rect;

use crate::core::desk::{Channel, DeskButton, StripKind};

/// A strip of the desk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelLayout {
    /// Which.
    pub channel: Channel,
    /// Its box, the border included (a click selects it, the wheel moves
    /// its fader).
    pub frame: Rect,
    /// The scribble strip with its name.
    pub name: Rect,
    /// The lines under the name.
    pub info: Rect,
    /// The tall section: the scale, the fader and the meters.
    pub section: Rect,
    /// The scale's labels, right-aligned (3 cells; the next is the tick).
    pub scale: Rect,
    /// The fader's travel: the cap's cells over the section's rows.
    pub track: Option<Rect>,
    /// The meters' bars, one per channel. A fader strip's span the fader's
    /// rows, on its scale; the others' have their clip lamp (or their
    /// label) in the row above.
    pub meters: Vec<Rect>,
    /// The row under the section: the level.
    pub value: Rect,
    /// The row under that.
    pub foot: Rect,
    /// The buttons.
    pub buttons: Vec<(Rect, DeskButton)>,
}

/// The desk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeskLayout {
    /// The strips shown, left to right, the master last.
    pub channels: Vec<ChannelLayout>,
}

impl DeskLayout {
    /// `channel`'s strip, if it is shown.
    pub fn channel(&self, channel: Channel) -> Option<&ChannelLayout> {
        self.channels.iter().find(|c| c.channel == channel)
    }
}

/// A strip's widths, border included: on a narrow desk, the least it grows
/// from, and the widest.
fn widths(channel: Channel) -> (u16, u16, u16) {
    match channel {
        Channel::Strip(_) => (12, 12, 16),
        Channel::Mic => (18, 18, 22),
        Channel::Main => (21, 26, 36),
    }
}

/// The gap between strips, and before the master.
const GAP: u16 = 1;
const MASTER_GAP: u16 = 3;

/// Info lines each strip wants under its name.
fn info_rows(channel: Channel) -> u16 {
    match channel {
        Channel::Strip(_) => 3,
        Channel::Mic => 6,
        Channel::Main => 4,
    }
}

/// The fewest rows a tall section is drawn with.
const MIN_SECTION: u16 = 4;

fn gaps(set: &[Channel]) -> u16 {
    let n = set.len() as u16;
    if n <= 1 {
        0
    } else {
        GAP * (n - 2) + MASTER_GAP
    }
}

/// The desk inside the list pane's border.
pub fn compute(inner: Rect) -> DeskLayout {
    let room = inner.width.saturating_sub(2);
    let all = Channel::ALL.to_vec();
    let sets: [Vec<Channel>; 3] = [
        all.clone(),
        vec![
            Channel::Strip(StripKind::Player),
            Channel::Strip(StripKind::Preview),
            Channel::Main,
        ],
        vec![Channel::Strip(StripKind::Player), Channel::Main],
    ];
    // The first set that fits: at its widths growing into the room, or at
    // the narrow ones.
    let mut chosen: Option<(Vec<Channel>, Vec<u16>)> = None;
    for set in &sets {
        let least: u16 = set.iter().map(|&c| widths(c).1).sum::<u16>() + gaps(set);
        let most: u16 = set.iter().map(|&c| widths(c).2).sum::<u16>() + gaps(set);
        if room >= least {
            let share = (f64::from(room - least) / f64::from((most - least).max(1))).min(1.0);
            let w = set
                .iter()
                .map(|&c| {
                    let (_, lo, hi) = widths(c);
                    lo + (f64::from(hi - lo) * share).floor() as u16
                })
                .collect();
            chosen = Some((set.clone(), w));
            break;
        }
        let tight: u16 = set.iter().map(|&c| widths(c).0).sum::<u16>() + gaps(set);
        if room >= tight {
            chosen = Some((set.clone(), set.iter().map(|&c| widths(c).0).collect()));
            break;
        }
    }
    let (set, w) = chosen.unwrap_or_else(|| (vec![Channel::Main], vec![room]));
    let total: u16 = w.iter().sum::<u16>() + gaps(&set);
    let lines = set.iter().map(|&c| info_rows(c)).max().unwrap_or(0);
    let mut x = inner.x + 1 + room.saturating_sub(total) / 2;
    let channels = set
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (&channel, &width))| {
            if i > 0 {
                x += if channel == Channel::Main {
                    MASTER_GAP
                } else {
                    GAP
                };
            }
            let frame = Rect::new(
                x,
                inner.y,
                width.min(inner.right().saturating_sub(x)),
                inner.height,
            );
            x += width;
            strip(channel, frame, lines)
        })
        .collect();
    DeskLayout { channels }
}

/// One strip in `frame`, its tall section under `lines` rows of lines (the
/// most any strip shown wants, so the sections line up as on a desk).
fn strip(channel: Channel, frame: Rect, lines: u16) -> ChannelLayout {
    let iw = frame.width.saturating_sub(2);
    let ih = frame.height.saturating_sub(2);
    let (x0, y0) = (frame.x + 1, frame.y + 1);
    let foot_rows = if ih >= 1 + 2 + MIN_SECTION { 2 } else { 0 };
    let info = lines.min(ih.saturating_sub(1 + foot_rows + MIN_SECTION));
    // A blank row under the lines, where there is room for it.
    let spacer = u16::from(ih >= 1 + info + 1 + foot_rows + 10);
    let section_y = y0 + 1 + info + spacer;
    let section_h = ih.saturating_sub(1 + info + spacer + foot_rows);
    let row = |y: u16| Rect::new(x0, y, iw, 1);
    let name = row(y0).intersection(frame);
    let info_rect = Rect::new(x0, y0 + 1, iw, info.min(info_rows(channel)));
    let section = Rect::new(x0, section_y, iw, section_h);
    let value = Rect::new(x0, section_y + section_h, iw, foot_rows.min(1));
    let foot = Rect::new(
        x0,
        section_y + section_h + 1,
        iw,
        foot_rows.saturating_sub(1),
    );
    let mut buttons = Vec::new();
    let mut track = None;
    let mut meters = Vec::new();
    let scale;
    match channel {
        Channel::Strip(kind) => {
            // The scale (3 and a tick), the travel (3), a gap and the two
            // meters, with a gap between where there is room.
            let apart = iw >= 11;
            let group = 4 + 3 + 1 + if apart { 3 } else { 2 };
            let gx = x0 + iw.saturating_sub(group) / 2;
            scale = Rect::new(gx, section.y, 3, section.height);
            track = Some(Rect::new(gx + 4, section.y, 3, section.height));
            // The meters on the fader's rows: one ruler for both.
            let mx = gx + 8;
            let bars = Rect::new(0, section.y, 1, section.height);
            meters.push(Rect { x: mx, ..bars });
            meters.push(Rect {
                x: mx + if apart { 2 } else { 1 },
                ..bars
            });
            if foot.height > 0 {
                let bx = x0 + iw.saturating_sub(7) / 2;
                buttons.push((Rect::new(bx, foot.y, 3, 1), DeskButton::Mute(kind)));
                buttons.push((Rect::new(bx + 4, foot.y, 3, 1), DeskButton::Solo(kind)));
            }
        }
        Channel::Mic => {
            // The scale and the two meters under their labels "IN VOICE".
            let group = 4 + 8;
            let gx = x0 + iw.saturating_sub(group) / 2;
            scale = Rect::new(gx, section.y + 1, 3, section.height.saturating_sub(1));
            let bars = Rect::new(0, section.y + 1, 2, section.height.saturating_sub(1));
            meters.push(Rect { x: gx + 4, ..bars });
            meters.push(Rect { x: gx + 7, ..bars });
            // The echo canceller's button: the third line.
            if info_rect.height >= 3 {
                buttons.push((Rect::new(x0, y0 + 3, 8.min(iw), 1), DeskButton::Echo));
            }
        }
        Channel::Main => {
            // The scale, two wide bars under their clip lamps, their names
            // under them.
            let bar = ((iw.saturating_sub(4 + 2 + 3)) / 2).clamp(2, 6);
            let group = 4 + 2 + bar * 2 + 3;
            let gx = x0 + iw.saturating_sub(group) / 2;
            let h = section.height.saturating_sub(2);
            scale = Rect::new(gx, section.y + 1, 3, h);
            let bars = Rect::new(0, section.y + 1, bar, h);
            meters.push(Rect { x: gx + 6, ..bars });
            meters.push(Rect {
                x: gx + 6 + bar + 3,
                ..bars
            });
        }
    }
    ChannelLayout {
        channel,
        frame,
        name,
        info: info_rect,
        section,
        scale,
        track,
        meters,
        value,
        foot,
        buttons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside(inner: Rect, outer: Rect) -> bool {
        inner.width == 0
            || inner.height == 0
            || (inner.x >= outer.x
                && inner.y >= outer.y
                && inner.right() <= outer.right()
                && inner.bottom() <= outer.bottom())
    }

    #[test]
    fn every_strip_fits_and_its_parts_are_inside_it() {
        for (w, h) in [(118, 32), (78, 20), (158, 46), (62, 18), (38, 8)] {
            let inner = Rect::new(1, 1, w, h);
            let desk = compute(inner);
            assert!(!desk.channels.is_empty());
            assert_eq!(desk.channels.last().unwrap().channel, Channel::Main);
            for c in &desk.channels {
                assert!(inside(c.frame, inner), "{w}×{h} {c:?}");
                let body = Rect::new(
                    c.frame.x + 1,
                    c.frame.y + 1,
                    c.frame.width - 2,
                    c.frame.height - 2,
                );
                for r in [c.name, c.info, c.section, c.scale, c.value, c.foot]
                    .into_iter()
                    .chain(c.track)
                    .chain(c.meters.iter().copied())
                    .chain(c.buttons.iter().map(|b| b.0))
                {
                    assert!(
                        inside(r, body),
                        "{w}×{h} {:?}: {r:?} in {body:?}",
                        c.channel
                    );
                }
            }
            for pair in desk.channels.windows(2) {
                assert!(pair[0].frame.right() < pair[1].frame.x, "{w}×{h} overlap");
            }
        }
    }

    #[test]
    fn a_wide_desk_has_every_strip_and_a_tall_fader() {
        let desk = compute(Rect::new(1, 1, 118, 32));
        assert_eq!(desk.channels.len(), 4);
        let player = desk.channel(Channel::Strip(StripKind::Player)).unwrap();
        let track = player.track.unwrap();
        assert!(track.height >= 20, "{track:?}");
        assert_eq!(player.buttons.len(), 2);
        let mic = desk.channel(Channel::Mic).unwrap();
        assert!(mic.track.is_none());
        assert!(mic.buttons.iter().any(|b| b.1 == DeskButton::Echo));
        // Narrow: the microphone goes first.
        let desk = compute(Rect::new(1, 1, 62, 18));
        assert!(desk.channel(Channel::Mic).is_none());
        assert!(desk.channel(Channel::Strip(StripKind::Player)).is_some());
    }
}
