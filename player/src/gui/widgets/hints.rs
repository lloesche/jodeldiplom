//! The keys a screen is moved about with ([`hints`]): each key's cap
//! ([`super::key`]) beside what it does.

use iced::widget::{row, text, Row};
use iced::{Alignment, Element, Widget as _};

use super::key;
use crate::gui::look::{self, size, space};

/// The keys `hints` (each its name and what it does): a cap and its
/// words, side by side (no name: the words alone, what to do).
pub fn hints<'a, Message: 'a>(hints: &[(&str, &str)]) -> Row<Element<'a, Message>> {
    hints
        .iter()
        .fold(Row::new().spacing(space::L), |all, (name, what)| {
            let words = text((*what).to_owned())
                .size(size::CAPTION)
                .font(look::font::body())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None);
            all.push(if name.is_empty() {
                words.boxed()
            } else {
                row![key(*name), words]
                    .spacing(space::S)
                    .align_y(Alignment::Center)
                    .boxed()
            })
        })
        .align_y(Alignment::Center)
}
