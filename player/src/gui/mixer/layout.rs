//! The Mixer tab's desk in points: the strips side by side, the master at
//! the right, as on a desk; the player's stems, narrower, beside it under
//! a bracket that says whether they or the file are heard.
//!
//! ```text
//!              ── ● STEMS heard ──
//! ╭────────╮ ╭─────╮╭─────╮      ╭─────────╮ ╭────────╮   ╭──────────╮
//! │ PLAYER │ │DRUMS││BASS │  …   │ PREVIEW │ │  MIC   │   │   MAIN   │
//! ╰────────╯ ╰─────╯╰─────╯      ╰─────────╯ ╰────────╯   ╰──────────╯
//! ```
//!
//! Each strip takes the desk's height less its edges (a stem's less the
//! bracket's too); strips grow from their least widths up to their
//! widest, the desk centred. Which strips show, and their widths, are the core's rule
//! ([`strips_shown`]), the same the arrow keys go by: the stems' while the
//! player has stems on the desk (its input has lanes); where it is too
//! narrow, the microphone goes first, then the karaoke game's effects'
//! strip, then the preview, then the stems.

use iced::{Rectangle, Size};

use crate::core::desk::shown::{gap, gaps, room, strips_shown, widths, Shown, Widths, EDGE};
use crate::core::desk::Channel;

/// The bracket's height over the stems' strips, in points.
pub const BRACKET: f32 = 24.0;

/// A strip of the desk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strip {
    /// Which.
    pub channel: Channel,
    /// Its box, in points from the desk's top left.
    pub frame: Rectangle,
}

/// The desk.
#[derive(Clone, Debug, PartialEq)]
pub struct Desk {
    /// The strips shown, left to right, the master last.
    pub strips: Vec<Strip>,
    /// The bracket over the stems' strips, while they show.
    pub stems: Option<Rectangle>,
}

impl Desk {
    /// `channel`'s strip, if it is shown.
    pub fn strip(&self, channel: Channel) -> Option<&Strip> {
        self.strips.iter().find(|s| s.channel == channel)
    }

    /// The channels shown, left to right.
    pub fn channels(&self) -> Vec<Channel> {
        self.strips.iter().map(|s| s.channel).collect()
    }
}

/// The desk in `size` points; `stems`: the player has its stems on the
/// desk. The strips shown ([`strips_shown`]), from their least widths
/// growing into the room, or at their narrow ones; whole points.
pub fn compute(size: Size, stems: bool) -> Desk {
    let room = room(size.width);
    let Shown {
        strips: set,
        widths: at,
    } = strips_shown(size.width, stems);
    let tight = at.tight();
    let w: Vec<f32> = match at {
        Widths::Least => {
            let sum = |which: fn((f32, f32, f32)) -> f32| -> f32 {
                set.iter().map(|&c| which(widths(c))).sum::<f32>() + gaps(&set, false)
            };
            let (least, most) = (sum(|w| w.1), sum(|w| w.2));
            let share = ((room - least) / (most - least).max(1.0)).min(1.0);
            set.iter()
                .map(|&c| {
                    let (_, lo, hi) = widths(c);
                    lo + ((hi - lo) * share).floor()
                })
                .collect()
        }
        Widths::Narrow => set.iter().map(|&c| widths(c).0).collect(),
        Widths::Room => vec![room.floor()],
    };
    let total: f32 = w.iter().sum::<f32>() + gaps(&set, tight);
    let mut x = EDGE + ((room - total).max(0.0) / 2.0).floor();
    let strips: Vec<Strip> = set
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (&channel, &width))| {
            if i > 0 {
                x += gap(set[i - 1], channel, tight);
            }
            let width = width.min((size.width - x).max(0.0));
            // The stems' strips under their bracket.
            let top = EDGE + if channel.is_part() { BRACKET } else { 0.0 };
            let frame = Rectangle::new(
                iced::Point::new(x, top),
                Size::new(width, (size.height - EDGE - top).max(0.0)),
            );
            x += width;
            Strip { channel, frame }
        })
        .collect();
    let bracket = {
        let mut parts = strips
            .iter()
            .filter(|s| s.channel.is_part())
            .map(|s| s.frame);
        let first = parts.next();
        let last = parts.next_back().or(first);
        first.zip(last).map(|(a, b)| {
            Rectangle::new(
                iced::Point::new(a.x, EDGE),
                Size::new(b.x + b.width - a.x, BRACKET),
            )
        })
    };
    Desk {
        strips,
        stems: bracket,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::desk::StripKind;

    fn desk(w: f32, h: f32, stems: bool) -> Desk {
        compute(Size::new(w, h), stems)
    }

    fn inside(a: Rectangle, size: Size) -> bool {
        a.x >= 0.0 && a.y >= 0.0 && a.x + a.width <= size.width && a.y + a.height <= size.height
    }

    #[test]
    fn every_strip_fits_side_by_side() {
        for (w, h, stems) in [
            (944.0, 600.0, true),
            (944.0, 600.0, false),
            (624.0, 380.0, true),
            (1264.0, 900.0, true),
            (496.0, 340.0, true),
            (496.0, 340.0, false),
            (304.0, 150.0, true),
        ] {
            let size = Size::new(w, h);
            let d = compute(size, stems);
            assert!(!d.strips.is_empty());
            assert_eq!(d.strips.last().unwrap().channel, Channel::Main);
            for s in &d.strips {
                assert!(inside(s.frame, size), "{w}×{h} {s:?}");
                assert_eq!(s.frame.x, s.frame.x.round(), "whole points");
                assert_eq!(s.frame.width, s.frame.width.round(), "whole points");
            }
            for pair in d.strips.windows(2) {
                // The stems side by side (touching); the rest a gap apart.
                let side_by_side = pair[0].channel.is_part() && pair[1].channel.is_part();
                let apart = if side_by_side { 0.0 } else { 8.0 };
                assert!(
                    pair[0].frame.x + pair[0].frame.width + apart <= pair[1].frame.x,
                    "{w}×{h} overlap"
                );
            }
        }
    }

    #[test]
    fn a_narrow_desk_loses_the_microphone_first() {
        // Wide: every strip, the effects' too (it takes no room).
        let d = desk(944.0, 600.0, false);
        assert_eq!(d.strips.len(), 5);
        assert!(d.stems.is_none());
        let d = desk(496.0, 340.0, false);
        assert!(d.strip(Channel::Mic).is_none());
        assert!(d.strip(Channel::Strip(StripKind::Player)).is_some());
    }

    #[test]
    fn the_stems_sit_beside_the_player_under_their_bracket() {
        // Wide enough for all, the effects' strip too.
        assert_eq!(desk(1360.0, 600.0, true).channels(), Channel::ALL);
        // At the usual widths too (a window 120 cells wide): karaoke with
        // the stems has its effects' strip.
        let d = desk(944.0, 600.0, true);
        assert_eq!(d.channels(), Channel::ALL);
        let player = d.strip(Channel::Strip(StripKind::Player)).unwrap();
        let bracket = d.stems.expect("the bracket");
        for kind in StripKind::PARTS {
            let s = d.strip(Channel::Strip(kind)).unwrap().frame;
            // Narrower, under the bracket, down to the player's bottom.
            assert!(s.width < player.frame.width, "{kind:?}");
            assert_eq!(s.y, bracket.y + BRACKET);
            assert_eq!(s.y + s.height, player.frame.y + player.frame.height);
            assert!(s.x >= bracket.x && s.x + s.width <= bracket.x + bracket.width);
        }
        // Narrower: the microphone, then the preview go; the stems stay
        // longer, the player and the master always.
        let names = |w| desk(w, 400.0, true).channels();
        assert!(!names(768.0).contains(&Channel::Mic));
        assert!(names(768.0).contains(&Channel::Strip(StripKind::Preview)));
        // An 80-cell window (624 points of desk): all but the microphone,
        // the master's lamps whole.
        let at80 = desk(624.0, 380.0, true);
        let shown = at80.channels();
        assert!(
            shown.contains(&Channel::Strip(StripKind::Preview)),
            "{shown:?}"
        );
        assert!(shown.contains(&Channel::Strip(StripKind::Vocals)));
        let main = at80.strip(Channel::Main).unwrap();
        assert!(main.frame.width >= 192.0, "{:?}", main.frame);
        let tight = names(560.0);
        assert!(!tight.contains(&Channel::Strip(StripKind::Preview)));
        assert!(tight.contains(&Channel::Strip(StripKind::Vocals)));
        let tighter = names(400.0);
        assert_eq!(tighter, [Channel::Strip(StripKind::Player), Channel::Main]);
    }
}
