//! The settings ([`crate::view::settings`]) in iced's widgets: a card for
//! each section, in two columns where the window is wide; each row its
//! label, what it is set to in words and its control (a switch, a check,
//! a segmented choice, a value that opens a list); the device list or the
//! data folder's list under its row while open; the notes under the
//! cards, and the keys at the top.
//!
//! A click on a row is the click the old screen's value took (what enter
//! does on it); a click on one of a choice's segments steps the row to it;
//! a click on a device or a folder chooses or opens it. The keys work as
//! well, through the window, and the row or entry they are on is kept in
//! view ([`reveal`]).

use iced::widget::{column, container, responsive, row, scrollable, text, Column, Space};
use iced::{Alignment, Background, Border, Element, Fill, Length, Padding, Size, Widget as _};

use crate::core::{Action, State};
use crate::gui::look::{self, radius, size, space};
use crate::gui::widgets::{
    self, check, hints, item, panel, progress, segment, segmented, switch, Reveal,
};
use crate::gui::window::Message;
use crate::view::settings::{
    self, Control, Entry, List, Mark, Row, Section, Settings, Tone, TITLE,
};

/// The narrowest window whose cards go in two columns, in points.
const TWO_COLUMNS: f32 = 960.0;

/// The widest the cards go, in points.
const WIDEST: f32 = 1160.0;

/// The room above the header, in points (as the help's, so the titles
/// stand in one place).
const TOP: f32 = crate::view::help::measure::TOP;
/// The header's height, as the help's.
const HEADER: f32 = crate::view::help::measure::HEADER;

/// How far a row that belongs to the one above is set in, in points.
const UNDER: f32 = 28.0;

/// The id of the scrollable the cards are in.
const SCROLLED: &str = "settings";

/// What keeps the row or the entry the keys are on in view, scrolled to
/// where it must be ([`Reveal`]).
pub fn reveal(state: &State) -> Reveal {
    let target = match (&state.settings.folders, state.settings.picking) {
        (Some(pick), _) => format!("entry-{}", pick.cursor),
        (None, true) => format!("entry-{}", state.settings.device_cursor),
        (None, false) => format!("setting-{}", state.settings.row),
    };
    Reveal::new(SCROLLED, target)
}

/// The settings for `state`.
pub fn view<'a>(state: &State) -> Element<'a, Message> {
    let settings = Settings::of(state);
    let page = responsive(move |size| page(&settings, size));
    container(page)
        .width(Fill)
        .height(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            ..container::Style::default()
        })
        .boxed()
}

/// The page in `size`: the header, the cards (scrolled where the window
/// is too short for them) and the notes.
fn page<'a>(settings: &Settings, size: Size) -> Element<'a, Message> {
    let width = (size.width - 2.0 * space::XXL).clamp(0.0, WIDEST);
    let header = row![
        text(TITLE)
            .size(size::TITLE)
            .font(look::font::title())
            .color(look::TEXT),
        Space::new().width(Fill),
        hints(&settings::hints()),
    ]
    .align_y(Alignment::Center)
    .height(HEADER);
    let cards: Element<'a, Message> = if size.width >= TWO_COLUMNS {
        // As many rows on the left as on the right, about: the sections
        // in order, down the left and then the right.
        let rows: usize = settings.sections.iter().map(|s| s.rows.len()).sum();
        let mut left = Column::new().spacing(space::XL).width(Fill);
        let mut right = Column::new().spacing(space::XL).width(Fill);
        let mut taken = 0;
        for section in &settings.sections {
            if taken * 2 < rows {
                left = left.push(card(section));
            } else {
                right = right.push(card(section));
            }
            taken += section.rows.len();
        }
        row![left, right].spacing(space::XL).boxed()
    } else {
        settings
            .sections
            .iter()
            .fold(Column::new().spacing(space::XL), |all, s| all.push(card(s)))
            .boxed()
    };
    let notes =
        settings
            .notes
            .iter()
            .fold(Column::new().spacing(space::XS), |all, (note, tone)| {
                let color = match tone {
                    Tone::Dim => look::TEXT_FAINT,
                    Tone::Warn => look::WARN,
                    Tone::Fail => look::FAIL,
                };
                all.push(
                    text(note.clone())
                        .size(size::CAPTION)
                        .font(look::font::body())
                        .color(color)
                        .boxed(),
                )
            });
    let body = column![cards, notes]
        .spacing(space::XL)
        .width(width)
        .padding(Padding {
            bottom: space::XL,
            ..Padding::ZERO
        });
    column![
        container(header).width(width).padding(Padding {
            top: TOP,
            bottom: space::M,
            ..Padding::ZERO
        }),
        scrollable(container(body).center_x(Fill))
            .id(SCROLLED)
            .height(Fill),
    ]
    .align_x(Alignment::Center)
    .width(Fill)
    .boxed()
}

/// A section: its title over a card of its rows.
fn card<'a>(section: &Section) -> Element<'a, Message> {
    let rows = section
        .rows
        .iter()
        .fold(Column::new().spacing(2.0), |all, r| all.push(setting(r)));
    column![
        text(section.title)
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM),
        panel(rows).padding(space::XS + 2.0).width(Fill),
    ]
    .spacing(space::S)
    .boxed()
}

/// A row: its label and words at the left, its control at the right (a
/// choice of more than two under them), its list under it.
fn setting<'a>(r: &Row) -> Element<'a, Message> {
    let mut words = Column::new().spacing(2.0).width(Fill).push(
        text(r.label)
            .size(size::BODY)
            .font(look::font::label())
            .color(look::TEXT),
    );
    if let Some(about) = &r.about {
        words = words.push(
            text(about.clone())
                .size(size::CAPTION)
                .font(look::font::body())
                .color(look::TEXT_DIM)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End),
        );
    }
    let wide = matches!(&r.control, Control::Choice { options, .. } if options.len() > 2);
    let line = row![words]
        .spacing(space::M)
        .align_y(Alignment::Center)
        .width(Fill);
    let line = if wide { line } else { line.push(control(r)) };
    let mut content = Column::new().spacing(space::S).push(line.boxed());
    if wide {
        content = content.push(control(r));
    }
    let indent = if r.under { UNDER } else { 0.0 };
    let made = item(
        container(content)
            .padding(Padding {
                left: indent,
                ..Padding::ZERO
            })
            .boxed(),
    )
    .selected(r.selected)
    .id(format!("setting-{}", r.index))
    .on_press(Message::Act(Action::SettingsClick(r.index, 0)));
    let mut all = Column::new().push(made.into());
    if let Some(list) = &r.list {
        all = all.push(entries(list, indent));
    }
    all.boxed()
}

/// A row's control.
fn control<'a>(r: &Row) -> Element<'a, Message> {
    match &r.control {
        Control::Switch { on, usable } => switch(*on, *usable),
        Control::Check { on, round } => check(*on, *round),
        Control::Choice { options, .. } => segmented(options.iter().enumerate().map(|(k, o)| {
            let step = r.control.step_to(k);
            segment(o.clone())
                .selected(step.is_none())
                .id(format!("setting-{}-{k}", r.index))
                .on_press_maybe(step.map(|s| Message::Act(Action::SettingsClick(r.index, s))))
        }))
        .boxed(),
        Control::Open(value) => container(
            container(
                row![
                    text(value.clone())
                        .size(size::BODY)
                        .font(look::font::body())
                        .color(look::TEXT)
                        .wrapping(text::Wrapping::None)
                        .ellipsis(text::Ellipsis::Start),
                    text("▾").size(size::CAPTION).color(look::TEXT_DIM),
                ]
                .spacing(space::S)
                .align_y(Alignment::Center),
            )
            .center_y(look::height::CHIP + 4.0)
            .padding(Padding::from([0.0, space::M]))
            .style(|_| container::Style {
                background: Some(Background::Color(look::RAISED)),
                border: Border {
                    color: look::EDGE,
                    width: look::line::EDGE,
                    radius: radius::CONTROL.into(),
                },
                ..container::Style::default()
            }),
        )
        // Half the row at most: a long path is shortened at its start.
        .width(Fill)
        .align_x(Alignment::End)
        .boxed(),
        Control::Moving { share, failed } => {
            let tone = if *failed { look::FAIL } else { look::ACCENT };
            row![
                container(progress(*share, tone)).width(Length::Fixed(160.0)),
                text(format!("{:.0} %", share * 100.0))
                    .size(size::CAPTION)
                    .font(look::font::numbers())
                    .color(tone),
            ]
            .spacing(space::S)
            .align_y(Alignment::Center)
            .boxed()
        }
    }
}

/// A list under its row, set in as the row is (and a little more): its
/// entries, each taking a click, and what shows while it is listed or
/// empty.
fn entries<'a>(list: &List, indent: f32) -> Element<'a, Message> {
    let mut all = Column::new().spacing(2.0).padding(Padding {
        left: indent + space::L,
        top: 2.0,
        ..Padding::ZERO
    });
    for e in &list.entries {
        all = all.push(entry(e));
    }
    if let Some(note) = &list.note {
        all = all.push(
            container(
                text(note.clone())
                    .size(size::LABEL)
                    .font(look::font::body())
                    .color(look::TEXT_DIM),
            )
            .padding(Padding::from([space::S, space::M]))
            .boxed(),
        );
    }
    all.boxed()
}

/// An entry of a list: its mark, name and detail.
fn entry<'a>(e: &Entry) -> Element<'a, Message> {
    let mark: Element<'a, Message> = match e.mark {
        Mark::Chosen => widgets::badge(Space::new(), 8.0, look::GOOD, look::GOOD).boxed(),
        Mark::Device => widgets::badge(
            Space::new(),
            8.0,
            iced::Color::TRANSPARENT,
            look::TEXT_FAINT,
        )
        .boxed(),
        Mark::Here => widgets::icon(widgets::Icon::Check)
            .size(14.0)
            .color(look::GOOD)
            .boxed(),
        Mark::Up => text("↑").size(size::BODY).color(look::TEXT_DIM).boxed(),
        Mark::Folder => text("▸").size(size::BODY).color(look::FOLDER).boxed(),
    };
    let name_color = match e.mark {
        Mark::Folder => look::FOLDER,
        Mark::Up => look::TEXT_DIM,
        _ => look::TEXT,
    };
    let mut line = row![
        container(mark).center_x(16.0),
        text(e.name.clone())
            .size(size::BODY)
            .font(if e.selected {
                look::font::label()
            } else {
                look::font::body()
            })
            .color(name_color)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::End),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center);
    if !e.detail.is_empty() {
        line = line.push(
            text(e.detail.clone())
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End)
                .boxed(),
        );
    }
    let action = match e.mark {
        Mark::Chosen | Mark::Device => Action::ChooseDevice(e.index),
        Mark::Here | Mark::Up | Mark::Folder => Action::FolderRow(e.index),
    };
    item(line.boxed())
        .padding(Padding::from([6.0, space::M]))
        .selected(e.selected)
        .id(format!("entry-{}", e.index))
        .on_press(Message::Act(action))
        .boxed()
}
