//! The setup screen ([`crate::view::setup`]) in iced's widgets: one panel
//! in the middle of the window, the steps marked as they go, the step
//! under way opened up with why it is needed and its bar, the whole's bar,
//! what went wrong when it did, and the choices as buttons (their keys
//! work as well, through the window).

use iced::widget::{column, container, responsive, row, scrollable, text, Column, Row, Space};
use iced::{Alignment, Background, Border, Color, Element, Fill, Shrink, Size, Widget as _};

use crate::gui::look::{self, radius, size, space};
use crate::gui::widgets::{self, badge, card, note, panel, progress, Icon};
use crate::gui::window::Message;
use crate::setup::Status;
use crate::view::setup::{
    percent, Choice, Current, Mark, Screen, StepLine, Stopped, KEPT, LOOKING, TITLE, WHOLE,
};

/// The panel's widest, in points.
const WIDTH: f32 = 680.0;

/// A step's mark, in points across.
const MARK: f32 = 22.0;

/// The setup screen for `status`.
pub fn view(status: &Status) -> Element<'static, Message> {
    let screen = Screen::of(status);
    responsive(move |window| page(&screen, window)).boxed()
}

/// The screen in a window of `window`'s size: the panel at most [`WIDTH`]
/// wide, in the middle; scrolled where the window is too short for it.
fn page(screen: &Screen, window: Size) -> Element<'static, Message> {
    let width = WIDTH.min(window.width - 2.0 * space::XL).max(0.0);
    let mut body = Column::new()
        .spacing(space::XL)
        .push(header(screen))
        .push(steps(screen))
        .push(whole(screen.whole));
    if let Some(stopped) = &screen.stopped {
        body = body.push(failure(stopped));
    }
    body = body.push(actions(screen));
    let panel = panel(body).padding(space::XXL).width(width);
    let page = container(panel).center_x(Fill).padding(space::XL);
    container(scrollable(page).height(Shrink))
        .center_y(Fill)
        .width(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            ..container::Style::default()
        })
        .boxed()
}

/// The player's name, the title and what setup is.
fn header(screen: &Screen) -> Element<'static, Message> {
    let icon = if screen.files {
        Icon::Download
    } else {
        Icon::Note
    };
    let mark = container(widgets::icon(icon).size(22.0).color(look::ACCENT))
        .center_x(44.0)
        .center_y(44.0)
        .style(|_| container::Style {
            background: Some(Background::Color(look::tint(
                look::ACCENT,
                look::SURFACE,
                0.1,
            ))),
            border: Border {
                radius: radius::PANEL.into(),
                ..Border::default()
            },
            ..container::Style::default()
        });
    let titles = column![
        text("Jodeldiplom")
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM),
        text(TITLE)
            .size(size::DISPLAY)
            .font(look::font::title())
            .color(look::TEXT),
    ]
    .spacing(space::XS);
    column![
        row![mark, titles]
            .spacing(space::L)
            .align_y(Alignment::Center),
        text(screen.intro())
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT_DIM),
    ]
    .spacing(space::L)
    .boxed()
}

/// The steps, the one under way opened up; above them, while the files
/// are looked for, that they are.
fn steps(screen: &Screen) -> Element<'static, Message> {
    let mut list = Column::<Element<'static, Message>>::new().spacing(space::XS);
    if screen.looking {
        list = list.push(
            container(
                text(LOOKING)
                    .size(size::BODY)
                    .font(look::font::body())
                    .color(look::TEXT_DIM),
            )
            .padding([space::XS, space::M])
            .boxed(),
        );
    }
    for (i, step) in screen.steps.iter().enumerate() {
        list = list.push(match screen.current.as_ref().filter(|c| c.index == i) {
            Some(current) => working(i, step, current),
            None => line(i, step),
        });
    }
    list.boxed()
}

/// A step's mark: a check when done, a cross when it failed, its number
/// otherwise (on the accent while under way).
fn mark(i: usize, mark: Mark) -> Element<'static, Message> {
    let number = |color| {
        text((i + 1).to_string())
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(color)
    };
    let clear = Color::TRANSPARENT;
    match mark {
        Mark::Done => badge(
            widgets::icon(Icon::Check).size(14.0).color(look::GOOD),
            MARK,
            look::tint(look::GOOD, look::SURFACE, 0.14),
            clear,
        )
        .boxed(),
        Mark::Failed => badge(
            widgets::icon(Icon::Cross).size(12.0).color(look::FAIL),
            MARK,
            look::tint(look::FAIL, look::SURFACE, 0.16),
            clear,
        )
        .boxed(),
        Mark::Working => badge(number(look::ACCENT_INK), MARK, look::ACCENT, clear).boxed(),
        Mark::ToDo => badge(number(look::TEXT_FAINT), MARK, clear, look::EDGE).boxed(),
    }
}

/// A step's title beside its mark.
fn title(step: &StepLine) -> Element<'static, Message> {
    let (color, font) = match step.mark {
        Mark::Done => (look::TEXT, look::font::body()),
        Mark::Working => (look::TEXT, look::font::label()),
        Mark::Failed => (look::FAIL, look::font::label()),
        Mark::ToDo => (look::TEXT_DIM, look::font::body()),
    };
    text(step.title.clone())
        .size(size::BODY)
        .font(font)
        .color(color)
        .width(Fill)
        .boxed()
}

/// A step not under way: its mark and title on a row.
fn line(i: usize, step: &StepLine) -> Element<'static, Message> {
    container(
        row![mark(i, step.mark), title(step)]
            .spacing(space::M)
            .align_y(Alignment::Center),
    )
    .padding([space::XS, space::M])
    .boxed()
}

/// The step under way, on a card: its title, why it is needed, and where
/// it is measured its bar, what it has done and its share.
fn working(i: usize, step: &StepLine, current: &Current) -> Element<'static, Message> {
    let mut body = Column::<Element<'static, Message>>::new()
        .spacing(space::S)
        .push(title(step))
        .push(
            text(current.why.clone())
                .size(size::LABEL)
                .font(look::font::body())
                .color(look::TEXT_DIM)
                .boxed(),
        );
    if let Some(share) = current.share {
        let figures = row![
            text(current.detail.clone())
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_DIM)
                .width(Fill),
            text(percent(share))
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::ACCENT),
        ]
        .align_y(Alignment::Center);
        body = body
            .push(Space::new().height(space::XS).boxed())
            .push(progress(share, look::ACCENT).boxed())
            .push(figures.boxed());
    }
    card(
        row![mark(i, step.mark), body]
            .spacing(space::M)
            .align_y(Alignment::Start),
    )
    .padding(space::M)
    .width(Fill)
    .boxed()
}

/// The whole's bar, its share beside its label.
fn whole(share: f64) -> Element<'static, Message> {
    column![
        row![
            text(WHOLE)
                .size(size::LABEL)
                .font(look::font::label())
                .color(look::TEXT_DIM)
                .width(Fill),
            text(percent(share))
                .size(size::LABEL)
                .font(look::font::numbers())
                .color(look::TEXT),
        ]
        .align_y(Alignment::Center),
        progress(share, look::GOOD),
    ]
    .spacing(space::S)
    .boxed()
}

/// What went wrong and what to do, set apart in the failure's tone.
fn failure(stopped: &Stopped) -> Element<'static, Message> {
    note(
        row![
            widgets::icon(Icon::Alert).size(18.0).color(look::FAIL),
            column![
                text(stopped.what.clone())
                    .size(size::BODY)
                    .font(look::font::label())
                    .color(look::FAIL),
                text(stopped.advice.clone())
                    .size(size::BODY)
                    .font(look::font::body())
                    .color(look::TEXT),
            ]
            .spacing(space::XS),
        ]
        .spacing(space::M),
        look::FAIL,
    )
    .width(Fill)
    .boxed()
}

/// The choices as buttons, at the right; while setup goes on, what
/// quitting keeps beside them.
fn actions(screen: &Screen) -> Element<'static, Message> {
    let mut bar = Row::new().spacing(space::S).align_y(Alignment::Center);
    bar = bar.push(if screen.stopped.is_none() && screen.files {
        text(KEPT)
            .size(size::CAPTION)
            .font(look::font::body())
            .color(look::TEXT_FAINT)
            .width(Fill)
            .boxed()
    } else {
        Space::new().width(Fill).boxed()
    });
    for offer in screen.offers() {
        let made = match offer.choice {
            Choice::Retry => widgets::primary(offer.label).leading(Icon::Retry),
            // Beside Try again it is the other way; alone, it stays quiet.
            Choice::Quit if screen.stopped.is_some() => widgets::secondary(offer.label),
            Choice::Quit => widgets::ghost(offer.label),
        };
        bar = bar.push(Element::from(
            made.key(offer.key).on_press(Message::Setup(offer.choice)),
        ));
    }
    bar.boxed()
}
