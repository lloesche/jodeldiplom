//! A link: words that go somewhere when clicked, underlined while the
//! pointer is over them (the pointer a hand there), on one line that ends
//! in `…` where the room is too narrow.

use iced::widget::text::{Ellipsis, Wrapping};
use iced::widget::{container, hover, mouse_area, text, Space};
use iced::{mouse, Background, Color, Element, Fill, Font, Padding, Widget as _};

/// Where a line's underline is, from its top: at the baseline, as a share
/// of the text's size (a line is 1.3 of the size high, the baseline in its
/// middle less half the size).
const BASELINE: f32 = 1.15;

/// `words` at `size` in `font` and `color` as a link: `on_press` when
/// clicked.
pub fn link<'a, Message: Clone + 'a>(
    words: impl Into<String>,
    size: f32,
    font: Font,
    color: Color,
    on_press: Message,
) -> Element<'a, Message> {
    let words = text(words.into())
        .size(size)
        .font(font)
        .color(color)
        .wrapping(Wrapping::None)
        .ellipsis(Ellipsis::End);
    let underline = container(
        container(Space::new().width(Fill).height(1.0)).style(move |_| container::Style {
            background: Some(Background::Color(color)),
            ..container::Style::default()
        }),
    )
    .padding(Padding {
        top: (size * BASELINE).round(),
        ..Padding::ZERO
    });
    mouse_area(hover(words, underline))
        .on_press(on_press)
        .interaction(mouse::Interaction::Pointer)
        .boxed()
}
