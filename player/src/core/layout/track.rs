//! A run of cells a knob moves along: a Karaoke slider's columns, a Mixer
//! tab fader's rows. The middle of its first cell is share 0 and of its
//! last cell share 1, where the knob is drawn for those shares, so the
//! pointer and the knob meet whatever the run's length.

use ratatui::layout::Rect;

/// A run of cells along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Track {
    /// Its first cell.
    pub start: u16,
    /// How many cells.
    pub len: u16,
}

impl Track {
    /// `rect`'s columns.
    pub fn columns(rect: Rect) -> Track {
        Track {
            start: rect.x,
            len: rect.width,
        }
    }

    /// `rect`'s rows (share 0 at the top).
    pub fn rows(rect: Rect) -> Track {
        Track {
            start: rect.y,
            len: rect.height,
        }
    }

    /// The share with the pointer `at` cells from the edge, 0 to 1 (none
    /// for a run of fewer than two cells).
    pub fn share_at(self, at: f64) -> Option<f64> {
        if self.len < 2 {
            return None;
        }
        let along = at - f64::from(self.start) - 0.5;
        Some((along / f64::from(self.len - 1)).clamp(0.0, 1.0))
    }

    /// Where the knob at `share` has its middle, in cells from the edge.
    pub fn centre(self, share: f64) -> f64 {
        f64::from(self.start) + 0.5 + share * f64::from(self.len.saturating_sub(1))
    }

    /// The cell the knob at `share` is drawn in.
    pub fn cell(self, share: f64) -> u16 {
        let cells = f64::from(self.len.saturating_sub(1));
        self.start + (share.clamp(0.0, 1.0) * cells).round() as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_is_where_the_pointer_puts_it() {
        let t = Track { start: 10, len: 21 };
        assert_eq!(t.share_at(10.5), Some(0.0));
        assert_eq!(t.share_at(30.5), Some(1.0));
        assert_eq!(t.share_at(0.0), Some(0.0));
        assert_eq!(t.share_at(99.0), Some(1.0));
        for s in [0.0, 0.25, 0.75, 1.0] {
            assert!((t.share_at(t.centre(s)).unwrap() - s).abs() < 1e-12);
        }
        assert_eq!(t.cell(0.0), 10);
        assert_eq!(t.cell(0.25), 15);
        assert_eq!(t.cell(1.0), 30);
        assert_eq!(Track { start: 3, len: 1 }.share_at(3.5), None);
    }
}
