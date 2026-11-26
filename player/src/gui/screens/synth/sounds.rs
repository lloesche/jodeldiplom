//! The Synth tab's sounds: each moment's name over its sounds, a row a
//! sound (a dot where one of its patches is edited in the theme shown),
//! the selected one with the accent's bar; a click plays one.

use iced::widget::{container, row, scrollable, text, Column, Space};
use iced::{Alignment, Element, Fill, Padding, Widget as _};

use crate::core::synth_tab::{self, SynthAction};
use crate::core::{Action, State};
use crate::gui::look::{self, size, space};
use crate::gui::widgets;
use crate::gui::window::Message;
use crate::sfx::bench::SOUNDS;

/// The id of the scrollable the sounds are in.
pub const SCROLLED: &str = "synth-sounds";

/// The id of sound `k`'s row.
pub fn id(k: usize) -> String {
    format!("synth-sound-{k}")
}

/// The sounds by moment, in a scrollable panel.
pub fn view<'a>(state: &State) -> Element<'a, Message> {
    let theme = synth_tab::theme(state);
    let mut list = Column::<Element<'a, Message>>::new().spacing(2.0);
    let mut moment = None;
    for (k, s) in SOUNDS.iter().enumerate() {
        if moment != Some(s.moment) {
            moment = Some(s.moment);
            list = list.push(
                container(
                    text(s.moment.name().to_uppercase())
                        .size(size::CAPTION)
                        .font(look::font::label())
                        .color(look::TEXT_FAINT),
                )
                .padding(Padding {
                    top: if k == 0 { space::XS } else { space::M },
                    bottom: space::XS,
                    left: space::M,
                    right: 0.0,
                })
                .boxed(),
            );
        }
        let edited = (s.atoms.iter()).any(|&a| synth_tab::edited(state, theme, a).is_some());
        let dot: Element<'a, Message> = if edited {
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
        let selected = k == state.synth.sound;
        let name = text(s.name.clone())
            .size(size::BODY)
            .color(if selected { look::TEXT } else { look::TEXT_DIM })
            .wrapping(text::Wrapping::None)
            .width(Fill);
        let line = row![name, dot]
            .spacing(space::S)
            .align_y(Alignment::Center)
            .padding(Padding::from([0.0, space::M]));
        let r = widgets::list_row(line.boxed())
            .selected(selected)
            .on_press(Message::Act(Action::Synth(SynthAction::Pick(k))));
        list = list.push(container(r).id(id(k)).boxed());
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
