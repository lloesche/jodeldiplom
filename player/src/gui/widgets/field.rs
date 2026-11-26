//! A text field ([`field`]): iced's text input in the look, on the panel's
//! ground with a 1-point edge, the accent's ring while it has the keys, the
//! selection in a faint accent.

use iced::widget::text_input::{self, Status, TextInput};
use iced::{Background, Border};

use crate::gui::look::{self, radius, size, space};

/// A field showing `value` (`placeholder` while it is empty); it takes
/// text once it is given what to send ([`TextInput::on_input`]).
pub fn field<'a, Message: Clone + 'a>(
    placeholder: impl Into<String>,
    value: impl Into<String>,
) -> TextInput<'a, Message> {
    TextInput::new(placeholder.into(), value.into())
        .size(size::BODY)
        .font(look::font::body())
        .padding([space::S, space::M])
        .style(style)
}

/// The field's look by its status.
fn style(_theme: &iced::Theme, status: Status) -> text_input::Style {
    let (edge, width) = match status {
        Status::Focused { .. } => (look::ACCENT, look::line::FOCUS),
        Status::Hovered => (look::TEXT_FAINT, look::line::EDGE),
        Status::Active | Status::Disabled => (look::EDGE, look::line::EDGE),
    };
    text_input::Style {
        background: Background::Color(look::SURFACE),
        border: Border {
            color: edge,
            width,
            radius: radius::CONTROL.into(),
        },
        placeholder: look::TEXT_FAINT,
        value: if status == Status::Disabled {
            look::TEXT_DIM
        } else {
            look::TEXT
        },
        selection: look::alpha(look::ACCENT, 0.35),
    }
}
