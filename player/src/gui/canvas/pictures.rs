//! The pictures as the widgets ask for them: their cells left empty while
//! the frame is drawn, each kept for the canvas to paint after it, or to
//! place for the window (a cover).

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::view::karaoke::KaraokePicture;
use crate::view::lanes::WavePlot;
use crate::view::transition::RulerPlot;
use crate::view::{self, Pictures};

/// A picture to paint after the frame, or to place for the window.
#[derive(Clone)]
pub(super) enum Paint {
    /// A cover (the window draws it), and whether it starts under a line
    /// through its first row.
    Cover(Rect, bool),
    /// A lane of the Transition tab.
    Wave(Box<WavePlot>),
    /// A Transition block's bar numbers.
    Ruler(Box<RulerPlot>),
    /// Karaoke mode's picture, over its cells.
    Karaoke(Rect, Box<KaraokePicture>),
}

impl Paint {
    /// The cells it goes in.
    pub(super) fn rect(&self) -> Rect {
        match self {
            Paint::Cover(r, _) => *r,
            Paint::Wave(plot) => plot.area,
            Paint::Ruler(plot) => plot.area,
            Paint::Karaoke(area, _) => *area,
        }
    }
}

/// Collects the pictures while the widgets draw; their cells stay empty.
pub(super) struct PixelPictures {
    /// Whether there is a cover to show (else its placeholder's cells).
    pub(super) has_art: bool,
    /// The pictures, in the order asked for.
    pub(super) paints: Vec<Paint>,
}

impl Pictures for PixelPictures {
    fn cover(&mut self, frame: &mut Frame, area: Rect, below_line: bool) {
        if self.has_art {
            view::blank(frame, area);
            self.paints.push(Paint::Cover(area, below_line));
        } else {
            view::placeholder(frame, area);
        }
    }

    fn wave(&mut self, frame: &mut Frame, plot: &WavePlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Wave(Box::new(plot.clone())));
    }

    fn ruler(&mut self, frame: &mut Frame, plot: &RulerPlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Ruler(Box::new(plot.clone())));
    }

    fn karaoke(&mut self, frame: &mut Frame, area: Rect, picture: &KaraokePicture) {
        view::blank(frame, area);
        self.paints
            .push(Paint::Karaoke(area, Box::new(picture.clone())));
    }
}
