//! The Mixer tab's desk in cells: the strips side by side, the master at the
//! right, as on a desk; the player's stems, narrower, beside it under a
//! bracket that says whether they or the file are heard.
//!
//! ```text
//!              ╶───── ● STEMS heard ─────╴
//! ╭──────────╮ ╭───────╮ ╭───────╮       ╭──────────╮ ╭────────╮   ╭──────╮
//! │  PLAYER  │ │ DRUMS │ │ BASS  │  …    │ PREVIEW  │ │  MIC   │   │ MAIN │
//! │ source…  │ │ ● 0.0 │ │ ● 0.0 │       │ A −3.2   │ │ name…  │   │ …    │
//! │ +6 │  ▂▂ │ │ │  ▂▂ │ │ │  ▂▂ │       │ +6 │     │ │ 0 ┤ IN │   │      │
//! │  0▐█▌ ██ │ │▐█▌ ██ │ │▐█▌ ██ │       │  0▐█▌    │ │   ┤ ▅▅ │   │      │
//! │ −∞ │  ██ │ │ │  ██ │ │ │  ██ │       │ −∞ │     │ │   ┤ ▅▅ │   │      │
//! │  0.0 dB  │ │ 0.0dB │ │ 0.0dB │       │  0.0 dB  │ │ LISTEN │   │      │
//! │ [M] [S]  │ │[M] [S]│ │[M] [S]│       │ [M] [S]  │ │ PITCH… │   │      │
//! ╰──────────╯ ╰───────╯ ╰───────╯       ╰──────────╯ ╰────────╯   ╰──────╯
//! ```
//!
//! Each strip's tall section takes the height left; strips grow from their
//! least widths up to their widest, the desk centred. The stems' strips are
//! there while the player has stems on the desk (its input has lanes).
//! Where it is too narrow, the microphone goes first, then the preview,
//! then the stems.

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
    /// The bracket over the stems' strips (its row), while they show.
    pub stems: Option<Rect>,
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
        Channel::Strip(kind) if kind.is_part() => (7, 7, 10),
        // The preview without a scale of its own where it is narrow.
        Channel::Strip(StripKind::Preview) => (9, 12, 16),
        Channel::Strip(_) => (12, 12, 16),
        Channel::Mic => (18, 18, 22),
        // The master's lamps whole: "○ BIT-PERFECT  ○ LIMIT".
        Channel::Main => (24, 26, 36),
    }
}

/// The gap between strips, and before the master (one at the narrow
/// widths); the stems' strips stand side by side under their bracket.
const GAP: u16 = 1;
const MASTER_GAP: u16 = 3;

/// The gap before `next`, after `prev`.
fn gap(prev: Channel, next: Channel, tight: bool) -> u16 {
    match (prev, next) {
        (_, Channel::Main) if tight => GAP,
        (_, Channel::Main) => MASTER_GAP,
        (a, b) if a.is_part() && b.is_part() => 0,
        _ => GAP,
    }
}

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

fn gaps(set: &[Channel], tight: bool) -> u16 {
    set.windows(2).map(|w| gap(w[0], w[1], tight)).sum()
}

/// The desk inside the list pane's border; `stems`: the player has its
/// stems on the desk.
pub fn compute(inner: Rect, stems: bool) -> DeskLayout {
    let room = inner.width.saturating_sub(2);
    let player = Channel::Strip(StripKind::Player);
    let preview = Channel::Strip(StripKind::Preview);
    let parts: Vec<Channel> = if stems {
        StripKind::PARTS.map(Channel::Strip).to_vec()
    } else {
        Vec::new()
    };
    let with = |rest: &[Channel]| -> Vec<Channel> {
        std::iter::once(player)
            .chain(parts.iter().copied())
            .chain(rest.iter().copied())
            .collect()
    };
    let mut sets: Vec<Vec<Channel>> = vec![
        with(&[preview, Channel::Mic, Channel::Main]),
        with(&[preview, Channel::Main]),
    ];
    if stems {
        sets.push(with(&[Channel::Main]));
    }
    sets.push(vec![player, Channel::Main]);
    // The first set that fits: at its widths growing into the room, or at
    // the narrow ones.
    let mut chosen: Option<(Vec<Channel>, Vec<u16>, bool)> = None;
    for set in &sets {
        let least: u16 = set.iter().map(|&c| widths(c).1).sum::<u16>() + gaps(set, false);
        let most: u16 = set.iter().map(|&c| widths(c).2).sum::<u16>() + gaps(set, false);
        if room >= least {
            let share = (f64::from(room - least) / f64::from((most - least).max(1))).min(1.0);
            let w = set
                .iter()
                .map(|&c| {
                    let (_, lo, hi) = widths(c);
                    lo + (f64::from(hi - lo) * share).floor() as u16
                })
                .collect();
            chosen = Some((set.clone(), w, false));
            break;
        }
        let tight: u16 = set.iter().map(|&c| widths(c).0).sum::<u16>() + gaps(set, true);
        if room >= tight {
            let w = set.iter().map(|&c| widths(c).0).collect();
            chosen = Some((set.clone(), w, true));
            break;
        }
    }
    let (set, w, tight) = chosen.unwrap_or_else(|| (vec![Channel::Main], vec![room], true));
    let total: u16 = w.iter().sum::<u16>() + gaps(&set, tight);
    let lines = set.iter().map(|&c| info_rows(c)).max().unwrap_or(0);
    let mut x = inner.x + 1 + room.saturating_sub(total) / 2;
    let channels = set
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (&channel, &width))| {
            if i > 0 {
                x += gap(set[i - 1], channel, tight);
            }
            let frame = Rect::new(
                x,
                inner.y,
                width.min(inner.right().saturating_sub(x)),
                inner.height,
            );
            x += width;
            // The stems' strips a row down, under their bracket, their
            // lines one fewer: the tall sections still line up.
            if channel.is_part() && frame.height > 1 {
                let frame = Rect {
                    y: frame.y + 1,
                    height: frame.height - 1,
                    ..frame
                };
                strip(channel, frame, lines.saturating_sub(1))
            } else {
                strip(channel, frame, lines)
            }
        })
        .collect::<Vec<ChannelLayout>>();
    let bracket = {
        let mut parts = channels
            .iter()
            .filter(|c| c.channel.is_part())
            .map(|c| c.frame);
        let first = parts.next();
        let last = parts.next_back().or(first);
        first
            .zip(last)
            .map(|(a, b)| Rect::new(a.x, inner.y, b.right() - a.x, 1))
    };
    DeskLayout {
        channels,
        stems: bracket,
    }
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
        Channel::Strip(kind) if kind.is_part() || iw < 10 => {
            // The travel (3), a gap where there is room and the two meters
            // side by side: no scale of its own (the player's, beside it,
            // reads it).
            let apart = u16::from(iw >= 6);
            let gx = x0 + iw.saturating_sub(3 + apart + 2) / 2;
            scale = Rect::new(gx, section.y, 0, section.height);
            track = Some(Rect::new(gx, section.y, 3, section.height));
            let bars = Rect::new(0, section.y, 1, section.height);
            meters.push(Rect {
                x: gx + 3 + apart,
                ..bars
            });
            meters.push(Rect {
                x: gx + 4 + apart,
                ..bars
            });
            if foot.height > 0 {
                // `[M] [S]`, or `M S` on a narrow strip.
                let (w, apart) = if iw >= 7 { (3, 4) } else { (1, 2) };
                let bx = x0 + iw.saturating_sub(w + apart) / 2;
                buttons.push((Rect::new(bx, foot.y, w, 1), DeskButton::Mute(kind)));
                let solo = Rect::new(bx + apart, foot.y, w, 1);
                buttons.push((solo, DeskButton::Solo(kind)));
            }
        }
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
        for (w, h, stems) in [
            (118, 32, true),
            (118, 32, false),
            (78, 20, true),
            (158, 46, true),
            (62, 18, true),
            (62, 18, false),
            (38, 8, true),
        ] {
            let inner = Rect::new(1, 1, w, h);
            let desk = compute(inner, stems);
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
                // The stems side by side (touching); the rest a gap apart.
                let side_by_side = pair[0].channel.is_part() && pair[1].channel.is_part();
                let gap = u16::from(!side_by_side);
                assert!(
                    pair[0].frame.right() + gap <= pair[1].frame.x,
                    "{w}×{h} overlap"
                );
            }
        }
    }

    #[test]
    fn a_wide_desk_has_every_strip_and_a_tall_fader() {
        let desk = compute(Rect::new(1, 1, 118, 32), false);
        assert_eq!(desk.channels.len(), 4);
        assert!(desk.stems.is_none());
        let player = desk.channel(Channel::Strip(StripKind::Player)).unwrap();
        let track = player.track.unwrap();
        assert!(track.height >= 20, "{track:?}");
        assert_eq!(player.buttons.len(), 2);
        let mic = desk.channel(Channel::Mic).unwrap();
        assert!(mic.track.is_none());
        assert!(mic.buttons.iter().any(|b| b.1 == DeskButton::Echo));
        // Narrow: the microphone goes first.
        let desk = compute(Rect::new(1, 1, 62, 18), false);
        assert!(desk.channel(Channel::Mic).is_none());
        assert!(desk.channel(Channel::Strip(StripKind::Player)).is_some());
    }

    #[test]
    fn the_stems_sit_beside_the_player_under_their_bracket() {
        let desk = compute(Rect::new(1, 1, 118, 32), true);
        let order: Vec<Channel> = desk.channels.iter().map(|c| c.channel).collect();
        assert_eq!(order, Channel::ALL);
        let player = desk.channel(Channel::Strip(StripKind::Player)).unwrap();
        let bracket = desk.stems.expect("the bracket");
        for kind in StripKind::PARTS {
            let c = desk.channel(Channel::Strip(kind)).unwrap();
            // Narrower, a row down under the bracket, the tall sections
            // and the buttons in line with the player's.
            assert!(c.frame.width < player.frame.width, "{kind:?}");
            assert_eq!(c.frame.y, bracket.y + 1);
            assert!(c.frame.x >= bracket.x && c.frame.right() <= bracket.right());
            assert_eq!(
                c.section,
                Rect {
                    x: c.section.x,
                    width: c.section.width,
                    ..player.section
                }
            );
            assert_eq!(c.track.unwrap().height, player.track.unwrap().height);
            assert_eq!(c.buttons.len(), 2);
            assert_eq!(c.foot.y, player.foot.y);
        }
        // Narrower: the microphone, then the preview go; the stems stay
        // longer, the player and the master always.
        let names = |w| {
            compute(Rect::new(1, 1, w, 24), true)
                .channels
                .iter()
                .map(|c| c.channel)
                .collect::<Vec<_>>()
        };
        assert!(!names(96).contains(&Channel::Mic));
        assert!(names(96).contains(&Channel::Strip(StripKind::Preview)));
        // An 80-column terminal (78 inside the pane): all but the
        // microphone, the master's lamps whole.
        let at80 = compute(Rect::new(1, 1, 78, 22), true);
        let shown: Vec<Channel> = at80.channels.iter().map(|c| c.channel).collect();
        assert!(
            shown.contains(&Channel::Strip(StripKind::Preview)),
            "{shown:?}"
        );
        assert!(shown.contains(&Channel::Strip(StripKind::Vocals)));
        let main = at80.channel(Channel::Main).unwrap();
        assert!(main.frame.width >= 24, "{:?}", main.frame);
        let tight = names(70);
        assert!(!tight.contains(&Channel::Strip(StripKind::Preview)));
        assert!(tight.contains(&Channel::Strip(StripKind::Vocals)));
        let tighter = names(50);
        assert_eq!(tighter, [Channel::Strip(StripKind::Player), Channel::Main]);
    }
}
