//! A progress bar: a thin pill-shaped track, filled from the left in a
//! tone.

use iced::widget::{progress_bar, ProgressBar};
use iced::{Background, Border, Color};

use crate::gui::look::{self, height, radius};

/// `share` (0 to 1) of a bar across its width, filled in `tone` (the
/// accent for what is under way, [`look::GOOD`] for the whole).
pub fn progress<'a>(share: f64, tone: Color) -> ProgressBar<'a> {
    progress_bar(0.0..=1.0, share.clamp(0.0, 1.0) as f32)
        .girth(height::BAR)
        .style(move |_| progress_bar::Style {
            background: Background::Color(look::EDGE),
            bar: Background::Color(tone),
            border: Border {
                radius: radius::PILL.into(),
                ..Border::default()
            },
        })
}
