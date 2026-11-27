//! The Synth tab's sounds: each moment's name over its sounds, a row a
//! sound (a dot where one of its patches is edited in the theme shown),
//! the selected one with the accent's bar; a click plays one.

use iced::widget::{container, row, scrollable, text, Column, Space};
use iced::{Alignment, Element, Fill, Padding, Widget as _};

use crate::core::synth_tab::SynthAction;
use crate::core::Action;
use crate::gui::look::{self, size, space};
use crate::gui::widgets;
use crate::gui::window::Message;
use crate::view::synth::Synth;

/// The id of the scrollable the sounds are in.
pub const SCROLLED: &str = "synth-sounds";

/// The id of sound `k`'s row.
pub fn id(k: usize) -> String {
    format!("synth-sound-{k}")
}

/// The sounds by moment, in a scrollable panel.
pub fn view<'a>(m: &Synth) -> Element<'a, Message> {
    let mut list = Column::<Element<'a, Message>>::new().spacing(2.0);
    for (i, (moment, rows)) in m.moments.iter().enumerate() {
        list = list.push(
            container(
                text(moment.name().to_uppercase())
                    .size(size::CAPTION)
                    .font(look::font::label())
                    .color(look::TEXT_FAINT),
            )
            .padding(Padding {
                top: if i == 0 { space::XS } else { space::M },
                bottom: space::XS,
                left: space::M,
                right: 0.0,
            })
            .boxed(),
        );
        for s in rows {
            let dot: Element<'a, Message> = if s.edited {
                widgets::badge(
                    Space::new(),
                    6.0,
                    look::ACCENT,
                    look::alpha(look::ACCENT, 0.0),
                )
                .boxed()
            } else {
                Space::new().width(6.0).boxed()
            };
            let name = text(s.name)
                .size(size::BODY)
                .color(if s.selected {
                    look::TEXT
                } else {
                    look::TEXT_DIM
                })
                .wrapping(text::Wrapping::None)
                .width(Fill);
            let line = row![name, dot]
                .spacing(space::S)
                .align_y(Alignment::Center)
                .padding(Padding::from([0.0, space::M]));
            let r = widgets::list_row(line.boxed())
                .selected(s.selected)
                .on_press(Message::Act(Action::Synth(SynthAction::Pick(s.k))));
            list = list.push(container(r).id(id(s.k)).boxed());
        }
    }
    widgets::card(
        scrollable(list.padding(Padding::from([space::XS, 0.0])))
            .id(SCROLLED)
            .height(Fill),
    )
    .padding(space::XS)
    .width(Fill)
    .height(Fill)
    .boxed()
}
