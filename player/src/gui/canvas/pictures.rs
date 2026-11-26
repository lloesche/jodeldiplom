//! The pictures as the widgets ask for them: their cells left empty while
//! the frame is drawn, each kept for the canvas to paint after it.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::view::karaoke::KaraokePicture;
use crate::view::lanes::WavePlot;
use crate::view::transition::RulerPlot;
use crate::view::{self, Pictures, Plot};

/// A picture to paint after the frame.
#[derive(Clone)]
pub(super) enum Paint {
    /// A cover, and whether it starts under a line through its first row.
    Cover(Rect, bool),
    Spectrum(Plot),
    /// Text over a cover: the cover's rect and the text's.
    Caption(Rect, Rect),
    /// A lane of the Transition tab.
    Wave(Box<WavePlot>),
    /// A Transition block's bar numbers.
    Ruler(Box<RulerPlot>),
    /// Karaoke mode's picture.
    Karaoke(Box<KaraokePicture>),
}

impl Paint {
    /// The cells it goes in.
    pub(super) fn rect(&self) -> Rect {
        match self {
            Paint::Cover(r, _) | Paint::Caption(_, r) => *r,
            Paint::Spectrum(plot) => plot.area,
            Paint::Wave(plot) => plot.area,
            Paint::Ruler(plot) => plot.area,
            Paint::Karaoke(picture) => picture.area,
        }
    }
}

/// Collects the pictures while the widgets draw; their cells stay empty.
pub(super) struct PixelPictures {
    /// Whether there is a cover to paint (else its placeholder's cells).
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

    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Spectrum(plot.clone()));
    }

    fn wave(&mut self, frame: &mut Frame, plot: &WavePlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Wave(Box::new(plot.clone())));
    }

    fn ruler(&mut self, frame: &mut Frame, plot: &RulerPlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Ruler(Box::new(plot.clone())));
    }

    fn karaoke(&mut self, frame: &mut Frame, picture: &KaraokePicture) {
        view::blank(frame, picture.area);
        self.paints.push(Paint::Karaoke(Box::new(picture.clone())));
    }

    fn text_over_cover(&self) -> bool {
        self.has_art
    }

    fn caption(&mut self, cover: Rect, caption: Rect) {
        self.paints.push(Paint::Caption(cover, caption));
    }
}
