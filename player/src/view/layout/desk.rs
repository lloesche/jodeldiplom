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
//! Each strip takes the pane's height (a stem's a row less, under the
//! bracket); strips grow from their least widths up to their widest, the
//! desk centred. What is inside a strip is the window's to lay out
//! ([crate::gui::mixer]). Which strips show is the core's rule
//! ([`strips_shown`]): the stems' while the player has stems on the desk
//! (its input has lanes); where it is too narrow, the microphone goes
//! first, then the karaoke game's effects' strip, then the preview, then
//! the stems.

use ratatui::layout::Rect;

use crate::core::desk::shown::{gap, gaps, strips_shown, widths, Shown, Widths};
use crate::core::desk::Channel;

/// A strip of the desk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelLayout {
    /// Which.
    pub channel: Channel,
    /// Its box.
    pub frame: Rect,
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

/// The desk's room inside the list pane's border `inner`: a cell less
/// each side.
pub fn room(inner: Rect) -> u16 {
    inner.width.saturating_sub(2)
}

/// The desk inside the list pane's border; `stems`: the player has its
/// stems on the desk. The strips shown in its room ([`strips_shown`]),
/// from their least widths growing into it, or at their narrow ones.
pub fn compute(inner: Rect, stems: bool) -> DeskLayout {
    let room = room(inner);
    let Shown {
        strips: set,
        widths: at,
    } = strips_shown(room, stems);
    let tight = at.tight();
    let w: Vec<u16> = match at {
        Widths::Least => {
            let sum = |which: fn((u16, u16, u16)) -> u16| -> u16 {
                set.iter().map(|&c| which(widths(c))).sum::<u16>() + gaps(&set, false)
            };
            let (least, most) = (sum(|w| w.1), sum(|w| w.2));
            let share = (f64::from(room - least) / f64::from((most - least).max(1))).min(1.0);
            set.iter()
                .map(|&c| {
                    let (_, lo, hi) = widths(c);
                    lo + (f64::from(hi - lo) * share).floor() as u16
                })
                .collect()
        }
        Widths::Narrow => set.iter().map(|&c| widths(c).0).collect(),
        Widths::Room => vec![room],
    };
    let total: u16 = w.iter().sum::<u16>() + gaps(&set, tight);
    let mut x = inner.x + 1 + room.saturating_sub(total) / 2;
    let channels: Vec<ChannelLayout> = set
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (&channel, &width))| {
            if i > 0 {
                x += gap(set[i - 1], channel, tight);
            }
            let mut frame = Rect::new(
                x,
                inner.y,
                width.min(inner.right().saturating_sub(x)),
                inner.height,
            );
            x += width;
            // The stems' strips a row down, under their bracket.
            if channel.is_part() && frame.height > 1 {
                frame.y += 1;
                frame.height -= 1;
            }
            ChannelLayout { channel, frame }
        })
        .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::desk::StripKind;

    fn inside(inner: Rect, outer: Rect) -> bool {
        inner.x >= outer.x
            && inner.y >= outer.y
            && inner.right() <= outer.right()
            && inner.bottom() <= outer.bottom()
    }

    #[test]
    fn every_strip_fits_side_by_side() {
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
    fn a_narrow_desk_loses_the_microphone_first() {
        // Wide: every strip, the effects' too (it takes no room).
        let desk = compute(Rect::new(1, 1, 118, 32), false);
        assert_eq!(desk.channels.len(), 5);
        assert!(desk.stems.is_none());
        let desk = compute(Rect::new(1, 1, 62, 18), false);
        assert!(desk.channel(Channel::Mic).is_none());
        assert!(desk.channel(Channel::Strip(StripKind::Player)).is_some());
    }

    #[test]
    fn the_stems_sit_beside_the_player_under_their_bracket() {
        // Wide enough for all, the effects' strip too.
        let wide = compute(Rect::new(1, 1, 170, 32), true);
        let order: Vec<Channel> = wide.channels.iter().map(|c| c.channel).collect();
        assert_eq!(order, Channel::ALL);
        // At the usual widths too: karaoke with the stems has its effects'
        // strip.
        let desk = compute(Rect::new(1, 1, 118, 32), true);
        let order: Vec<Channel> = desk.channels.iter().map(|c| c.channel).collect();
        assert_eq!(order, Channel::ALL);
        let player = desk.channel(Channel::Strip(StripKind::Player)).unwrap();
        let bracket = desk.stems.expect("the bracket");
        for kind in StripKind::PARTS {
            let c = desk.channel(Channel::Strip(kind)).unwrap();
            // Narrower, a row down under the bracket, down to the player's
            // bottom.
            assert!(c.frame.width < player.frame.width, "{kind:?}");
            assert_eq!(c.frame.y, bracket.y + 1);
            assert_eq!(c.frame.bottom(), player.frame.bottom());
            assert!(c.frame.x >= bracket.x && c.frame.right() <= bracket.right());
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
        // An 80-column window (78 inside the pane): all but the
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
