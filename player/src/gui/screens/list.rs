//! The list pane ([`crate::view::list`]) in iced's widgets, over the
//! place the cell layout leaves for it: the tabs on its top edge (a
//! segmented control whose pill slides to the side shown), and on the
//! browser's or the queue's side a panel under them: where the browser
//! is (or which track of the queue plays), the rows in view, a note where
//! there is nothing to list, and the filter line at its foot. On a tab of
//! its own (Transition, Karaoke, the Mixer) only the tabs: what is under
//! them is the old canvas's, or the Karaoke tab's ([`super::karaoke`]).
//!
//! The rows are the core's page: only those in view are made, at a
//! fixed height, from the first in view; how many fit is what the window
//! tells the core ([`fit`]), so the keys page by what shows. A row's
//! press is the core's click ([`Action::ClickRow`]: two make a double
//! click), its right press queues it; a queue row's press holds the
//! pointer too ([`Grab::QueueRow`]): the track follows it to the row it
//! is over. With Ctrl held for a preview, the pointer over a browser row
//! selects it. The wheel scrolls the list ([`Message::Wheel`]).

use iced::widget::{
    column, container, mouse_area, rich_text, row, span, stack, text, Column, Row, Space,
};
use iced::{
    Alignment, Background, Border, Color, Element, Fill, Length, Padding, Point, Rectangle,
    Widget as _,
};
use jodel_playback::PlayState;

use crate::core::{Action, Fit, Readiness, Side, State};
use crate::gui::look::{self, height, radius, size, space};
use crate::gui::pointer::Grab;
use crate::gui::widgets::{self, list_row, Icon, Tab};
use crate::gui::window::Message;
use crate::view::layout::{self, Grid};
use crate::view::list::{
    Body, Browser, BrowserRow, Fact, FilterLine, Kind, Known, Note, Pane, Place, Queue, QueueRow,
    QUEUE_EMPTY, QUEUE_HINT,
};

/// The line under the tabs saying where the browser is, in points.
const PLACE: f32 = 30.0;
/// Under the last row, inside the panel's foot.
const FOOT: f32 = 6.0;
/// A row's content in from the panel's sides.
const INSET: f32 = 6.0;
/// The chip's column at a file row's right, in points: its chip ends at
/// the same place in every row.
const CHIP: f32 = 88.0;
/// The length's column, right of the chips: `1:23:45` in tabular figures.
const LENGTH: f32 = 48.0;
/// A row's icons.
const ICON: f32 = 16.0;
/// The icons of what is known of a queued track.
const FACT: f32 = 14.0;

/// Where the pane is: its size in points and a cell's (the tabs take the
/// top row of cells; the panel's edges run through the middle of the
/// border cells, where a tab's box of its own has its border).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// The pane's width.
    pub width: f32,
    /// The pane's height.
    pub height: f32,
    /// A cell's width and height.
    pub cell: (f32, f32),
}

/// The list's rows that fit a pane `height` points high, its tabs' row
/// `head` high: the core's page (the filter line takes the last while
/// the browser is filtered).
pub fn rows_in(height: f32, head: f32) -> u16 {
    let room = height - head / 2.0 - FOOT - top(head);
    (room / height::ROW).floor().clamp(0.0, f32::from(u16::MAX)) as u16
}

/// Where the first row starts, from the pane's top.
fn top(head: f32) -> f32 {
    head + space::XS / 2.0 + PLACE
}

/// What `grid` does to the keys and the pointer, a cell `cell` points:
/// the cell layout's ([`layout::fit`]), the list's page the rows this
/// pane shows.
pub fn fit(state: &State, grid: Grid, cell: (f32, f32)) -> Fit {
    let mut fit = layout::fit(state, grid);
    if fit.list_rows > 0 {
        let pane = layout::list_pane(grid);
        fit.list_rows = rows_in(f32::from(pane.height) * cell.1, cell.1);
    }
    fit
}

/// The pane for `state` at `place`.
pub fn view<'a>(state: &State, place: Room) -> Element<'a, Message> {
    let pane = Pane::of(state, place.width);
    let (cw, ch) = place.cell;
    let shown = pane.tabs.iter().position(|t| t.shown).unwrap_or(0);
    let tabs: Vec<Tab<Message>> = pane
        .tabs
        .iter()
        .map(|t| Tab {
            label: t.label.to_owned(),
            count: t.count.map(|n| n.to_string()),
            on_press: Message::Act(Action::ShowSide(t.side)),
        })
        .collect();
    let header = container(widgets::tabs(tabs, shown, ch))
        .padding(Padding::from([0.0, cw / 2.0 + space::S]))
        .height(ch);
    let (ground, body): (Element<'a, Message>, Element<'a, Message>) = match &pane.body {
        Body::Tab => (Space::new().boxed(), Space::new().boxed()),
        Body::Browser(b) => (ground(place), browser(state, b, place.width - place.cell.0)),
        Body::Queue(q) => (ground(place), queue(state, q)),
    };
    let body = container(body)
        .padding(Padding {
            top: space::XS / 2.0,
            right: cw / 2.0 + INSET,
            bottom: ch / 2.0 + FOOT,
            left: cw / 2.0 + INSET,
        })
        .width(Fill)
        .height(Fill);
    let pane = mouse_area(
        stack![ground, column![header, body]]
            .width(Fill)
            .height(Fill),
    );
    // The wheel scrolls the browser or the queue; over a tab of its own's
    // tabs it does nothing.
    match state.side {
        Side::Browser | Side::Queue => pane.on_scroll(Message::Wheel).boxed(),
        Side::Transition | Side::Karaoke | Side::Mixer => pane.boxed(),
    }
}

/// The panel under the tabs, its edges through the middle of the border
/// cells (the Karaoke tab's too).
pub(super) fn ground<'a>(place: Room) -> Element<'a, Message> {
    let (cw, ch) = place.cell;
    container(
        container(Space::new().width(Fill).height(Fill))
            .width(Fill)
            .height(Fill)
            .style(|_| container::Style {
                background: Some(Background::Color(look::SURFACE)),
                border: Border {
                    color: look::EDGE,
                    width: look::line::EDGE,
                    radius: radius::PANEL.into(),
                },
                ..container::Style::default()
            }),
    )
    .padding(Padding::from([ch / 2.0, cw / 2.0]))
    .width(Fill)
    .height(Fill)
    .boxed()
}

/// A line of the pane's own: an icon, then `content`, `trailing` at the
/// right.
fn line<'a>(
    icon: Icon,
    tone: Color,
    content: Element<'a, Message>,
    trailing: Option<Element<'a, Message>>,
) -> Element<'a, Message> {
    let mut parts = row![
        widgets::icon(icon).size(14.0).color(tone),
        container(content).width(Fill),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center);
    if let Some(trailing) = trailing {
        parts = parts.push(trailing);
    }
    container(parts)
        .padding(Padding::from([0.0, space::M]))
        .height(PLACE)
        .center_y(PLACE)
        .boxed()
}

/// The spinner's turn now, from the clock the frame is drawn at.
pub(super) fn turn(state: &State) -> f32 {
    let ms = state.drawn_ms.unwrap_or(state.now_ms);
    (ms % 1000) as f32 / 1000.0
}

/// A busy arc, turning.
fn busy<'a>(state: &State, tone: Color, size: f32) -> Element<'a, Message> {
    widgets::icon(Icon::Busy)
        .size(size)
        .color(tone)
        .turn(turn(state))
        .into()
}

/// `crumbs` that fit `room` points, the first ones going first (an `…`
/// in their place); measured by an average letter of the line's size
/// (iced's rich text does not shorten).
fn fitted(crumbs: &[String], room: f32) -> Vec<String> {
    const LETTER: f32 = 7.4;
    const BETWEEN: f32 = 22.0;
    let width = |parts: &[String]| -> f32 {
        parts
            .iter()
            .map(|p| p.chars().count() as f32 * LETTER + BETWEEN)
            .sum()
    };
    let mut from = 0;
    while from + 1 < crumbs.len()
        && width(&crumbs[from..]) + if from > 0 { BETWEEN } else { 0.0 } > room
    {
        from += 1;
    }
    let mut out = Vec::with_capacity(crumbs.len() - from + 1);
    if from > 0 {
        out.push("…".to_owned());
    }
    out.extend(crumbs[from..].iter().cloned());
    out
}

/// Where the browser is, `width` points across: its folder's parts, the
/// last bright; how many a search found, and whether it is still listing
/// or searching.
fn place_line<'a>(state: &State, place: &Place, width: f32) -> Element<'a, Message> {
    let mut spans = Vec::<text::Span<'a, ()>>::new();
    let busy_room = if place.found.is_some() || place.listing {
        150.0
    } else {
        0.0
    };
    let crumbs = fitted(&place.crumbs, width - 60.0 - busy_room);
    let last = crumbs.len().saturating_sub(1);
    for (i, crumb) in crumbs.into_iter().enumerate() {
        if i > 0 {
            spans.push(span("  ›  ").color(look::TEXT_FAINT));
        }
        let crumb = span(crumb);
        spans.push(if i == last {
            crumb.color(look::TEXT).font(look::font::label())
        } else {
            crumb.color(look::TEXT_DIM)
        });
    }
    let crumbs = rich_text(spans)
        .size(size::LABEL)
        .font(look::font::body())
        .wrapping(text::Wrapping::None)
        .width(Fill);
    let caption = |s: String| {
        text(s)
            .size(size::CAPTION)
            .font(look::font::body())
            .color(look::TEXT_DIM)
            .wrapping(text::Wrapping::None)
    };
    let status: Option<Element<'a, Message>> = match place.found {
        Some((n, true)) => Some(
            row![
                busy(state, look::ACCENT, 12.0),
                caption(format!("{n} found, searching…"))
            ]
            .spacing(space::XS)
            .align_y(Alignment::Center)
            .boxed(),
        ),
        Some((n, false)) => Some(caption(format!("{n} found below")).boxed()),
        None if place.listing => Some(
            row![busy(state, look::ACCENT, 12.0), caption("listing".into())]
                .spacing(space::XS)
                .align_y(Alignment::Center)
                .boxed(),
        ),
        None => None,
    };
    let icon = if place.found.is_some() {
        Icon::Search
    } else {
        Icon::Folder
    };
    line(icon, look::FOLDER, crumbs.boxed(), status)
}

/// A row's title in `tone`, shortened at its end.
fn title<'a>(title: String, tone: Color, bold: bool) -> Element<'a, Message> {
    text(title)
        .size(size::BODY)
        .font(if bold {
            look::font::label()
        } else {
            look::font::body()
        })
        .color(tone)
        .wrapping(text::Wrapping::None)
        .ellipsis(text::Ellipsis::End)
        .width(Fill)
        .boxed()
}

/// A file's chip and length, in their columns at the row's right.
fn known<'a>(known: &Known) -> Element<'a, Message> {
    let (chip, length): (Element<'a, Message>, String) = match known {
        Known::Nothing => return Space::new().width(0).boxed(),
        Known::Pending => (
            text("…")
                .size(size::CAPTION)
                .color(look::TEXT_FAINT)
                .boxed(),
            String::new(),
        ),
        Known::Read(chip, length) => (
            widgets::tag(
                chip.text.clone(),
                if chip.failed {
                    look::FAIL
                } else {
                    look::SOURCE
                },
            ),
            length.clone().unwrap_or_default(),
        ),
    };
    row![
        container(chip).width(CHIP).align_x(Alignment::End),
        text(length)
            .size(size::LABEL)
            .font(look::font::numbers())
            .color(look::TEXT_DIM)
            .wrapping(text::Wrapping::None)
            .width(LENGTH)
            .align_x(Alignment::End),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center)
    .boxed()
}

/// The browser's side: where it is, its rows, its note, its filter.
fn browser<'a>(state: &State, b: &Browser, width: f32) -> Element<'a, Message> {
    let preview = crate::core::preview::held(state);
    let mut rows = Column::<Element<'a, Message>>::new();
    for r in &b.rows {
        rows = rows.push(browser_row(r, preview));
    }
    if let Some(note) = &b.note {
        rows = rows.push(note_line(state, note));
    }
    let mut out = column![
        place_line(state, &b.place, width),
        container(rows).height(Fill)
    ];
    if let Some(filter) = &b.filter {
        out = out.push(filter_line(filter));
    }
    out.boxed()
}

/// A browser row.
fn browser_row<'a>(r: &BrowserRow, preview: bool) -> Element<'a, Message> {
    let (icon, tone) = match r.kind {
        Kind::Parent => (Icon::Up, look::TEXT_FAINT),
        Kind::Folder => (Icon::Folder, look::FOLDER),
        Kind::Playlist => (Icon::Playlist, look::FOLDER),
        Kind::File if r.playing => (Icon::Play, look::ACCENT),
        Kind::File => (Icon::Note, look::TEXT_FAINT),
        Kind::SearchBelow => (Icon::Search, look::ACCENT),
    };
    let lead = widgets::icon(icon).size(ICON).color(tone);
    let name_tone = if r.playing { look::ACCENT } else { look::TEXT };
    let (name, trailing): (Element<'a, Message>, Element<'a, Message>) = match r.kind {
        Kind::Parent => (
            title("..".into(), look::TEXT_DIM, false),
            text("up")
                .size(size::CAPTION)
                .color(look::TEXT_FAINT)
                .boxed(),
        ),
        Kind::Folder => (
            title(r.title.clone(), look::TEXT, false),
            widgets::icon(Icon::Chevron)
                .size(14.0)
                .color(look::TEXT_FAINT)
                .into(),
        ),
        Kind::Playlist => (
            title(r.title.clone(), look::TEXT, false),
            text("playlist")
                .size(size::CAPTION)
                .color(look::TEXT_FAINT)
                .boxed(),
        ),
        Kind::File => (
            title(r.title.clone(), name_tone, r.playing),
            known(&r.known),
        ),
        Kind::SearchBelow => (
            title(
                format!("Search all subfolders for “{}”", r.title),
                look::ACCENT,
                true,
            ),
            widgets::key("enter"),
        ),
    };
    let content = row![lead, name, trailing]
        .spacing(space::M)
        .align_y(Alignment::Center);
    let mut out = list_row(content.boxed())
        .selected(r.selected)
        .on_press(Message::Act(Action::ClickRow(r.index)))
        .on_right_press(Message::Act(Action::EnqueueRow(r.index)));
    if preview && !r.selected {
        out = out.on_move(Message::Act(Action::SelectRow(r.index)));
    }
    out.into()
}

/// What the browser says where there is nothing (else) to list.
fn note_line<'a>(state: &State, note: &Note) -> Element<'a, Message> {
    let (lead, tone): (Element<'a, Message>, Color) = match note {
        Note::Failed(_) => (
            widgets::icon(Icon::Alert)
                .size(ICON)
                .color(look::FAIL)
                .into(),
            look::FAIL,
        ),
        Note::Listing => (busy(state, look::ACCENT, ICON), look::TEXT_DIM),
        Note::NothingFound => (
            widgets::icon(Icon::Search)
                .size(ICON)
                .color(look::TEXT_FAINT)
                .into(),
            look::TEXT_DIM,
        ),
        Note::Empty => (
            widgets::icon(Icon::Folder)
                .size(ICON)
                .color(look::TEXT_FAINT)
                .into(),
            look::TEXT_DIM,
        ),
    };
    container(
        row![
            lead,
            text(note.text().to_owned())
                .size(size::BODY)
                .font(look::font::body())
                .color(tone)
        ]
        .spacing(space::M)
        .align_y(Alignment::Center),
    )
    .padding(Padding::from([space::M, space::M]))
    .boxed()
}

/// The filter line: what is typed (the caret while it is), and how many
/// rows it leaves.
fn filter_line<'a>(filter: &FilterLine) -> Element<'a, Message> {
    let typed: Element<'a, Message> = if filter.typed.is_empty() && filter.editing {
        text("type to filter")
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT_FAINT)
            .boxed()
    } else {
        text(filter.typed.clone())
            .size(size::BODY)
            .font(look::font::label())
            .color(look::TEXT)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::Start)
            .boxed()
    };
    let caret: Element<'a, Message> = if filter.editing {
        container(Space::new().width(2).height(16))
            .style(|_| container::Style {
                background: Some(Background::Color(look::ACCENT)),
                ..container::Style::default()
            })
            .boxed()
    } else {
        Space::new().width(0).boxed()
    };
    let editing = filter.editing;
    let field = container(
        row![
            widgets::icon(Icon::Search).size(14.0).color(look::ACCENT),
            row![typed, caret].spacing(1).align_y(Alignment::Center),
            Space::new().width(Fill),
            text(format!("{} shown", filter.shown))
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None),
        ]
        .spacing(space::S)
        .align_y(Alignment::Center),
    )
    .padding(Padding::from([0.0, space::M]))
    .center_y(height::ROW - space::XS)
    .width(Fill)
    .style(move |_| container::Style {
        background: Some(Background::Color(look::BG)),
        border: Border {
            color: if editing {
                look::alpha(look::ACCENT, 0.55)
            } else {
                look::EDGE
            },
            width: look::line::EDGE,
            radius: radius::CONTROL.into(),
        },
        ..container::Style::default()
    });
    container(field)
        .height(height::ROW)
        .align_y(Alignment::End)
        .boxed()
}

/// The queue's side: which track plays, the rows in view, or how to
/// fill it.
fn queue<'a>(state: &State, q: &Queue) -> Element<'a, Message> {
    let summary = line(
        Icon::Queue,
        look::TEXT_DIM,
        text(q.summary.clone())
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM)
            .boxed(),
        None,
    );
    if q.rows.is_empty() {
        let empty = column![
            widgets::icon(Icon::Queue)
                .size(32.0)
                .color(look::TEXT_FAINT),
            text(QUEUE_EMPTY)
                .size(size::TITLE)
                .font(look::font::title())
                .color(look::TEXT_DIM),
            text(QUEUE_HINT)
                .size(size::LABEL)
                .font(look::font::body())
                .color(look::TEXT_FAINT)
                .align_x(Alignment::Center)
                .width(Length::Fixed(380.0)),
        ]
        .spacing(space::S)
        .align_x(Alignment::Center);
        return column![summary, container(empty).center(Fill)].boxed();
    }
    let shown = q.rows.len();
    let first = q.rows.first().map_or(0, |r| r.index);
    let mut rows = Column::<Element<'a, Message>>::new();
    for r in &q.rows {
        rows = rows.push(queue_row(state, r, r.index - first, shown));
    }
    column![summary, container(rows).height(Fill)].boxed()
}

/// A queue row, `slot` rows below the first of the `shown` in view.
fn queue_row<'a>(state: &State, r: &QueueRow, slot: usize, shown: usize) -> Element<'a, Message> {
    let lead: Element<'a, Message> = match r.current {
        Some(play) => widgets::icon(match play {
            PlayState::Playing => Icon::Play,
            PlayState::Paused => Icon::Pause,
            PlayState::Stopped => Icon::Stop,
        })
        .size(ICON)
        .color(look::ACCENT)
        .into(),
        None => text((r.index + 1).to_string())
            .size(size::LABEL)
            .font(look::font::numbers())
            .color(look::TEXT_FAINT)
            .wrapping(text::Wrapping::None)
            .boxed(),
    };
    let lead = container(lead).width(28.0).align_x(Alignment::End);
    let tone = if r.current.is_some() {
        look::ACCENT
    } else {
        look::TEXT
    };
    let ready: Element<'a, Message> = match r.readiness {
        Readiness::Working => busy(state, look::ACCENT, FACT),
        Readiness::Waiting => {
            widgets::badge(Space::new(), 5.0, look::TEXT_FAINT, Color::TRANSPARENT).boxed()
        }
        Readiness::Done => Space::new().width(0).boxed(),
    };
    let mut facts = Row::<Element<'a, Message>>::new()
        .spacing(space::XS + 2.0)
        .align_y(Alignment::Center);
    for s in &r.facts {
        let icon = match s.fact {
            Fact::Key => Icon::Key,
            Fact::Stems => Icon::Stems,
            Fact::Tempo => Icon::Tempo,
            Fact::Lyrics => Icon::Lyrics,
            Fact::Cover => Icon::Cover,
        };
        let color = match (s.lit, s.heard) {
            (true, true) => look::WARN,
            (true, false) => look::ACCENT,
            (false, _) => look::alpha(look::TEXT_FAINT, 0.6),
        };
        facts = facts.push(widgets::icon(icon).size(FACT).color(color).into());
    }
    let content = row![
        lead,
        title(r.title.clone(), tone, r.current.is_some()),
        container(ready).width(FACT).center_x(FACT),
        facts,
        known(&r.known),
    ]
    .spacing(space::M)
    .align_y(Alignment::Center);
    list_row(content.boxed())
        .selected(r.selected)
        .on_press(Message::Act(Action::ClickRow(r.index)))
        .on_grab(move |bounds: Rectangle, at: Point| {
            let rows = Rectangle {
                x: bounds.x,
                y: bounds.y - slot as f32 * bounds.height,
                width: bounds.width,
                height: shown as f32 * bounds.height,
            };
            Message::Grab(
                Grab::QueueRow {
                    rows,
                    row: bounds.height,
                },
                at,
            )
        })
        .on_right_press(Message::Act(Action::EnqueueRow(r.index)))
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path too long for its line loses its first parts, an `…` in
    /// their place; the folder itself always stays.
    #[test]
    fn the_path_drops_its_first_parts_to_fit() {
        let crumbs: Vec<String> = ["F:", "Music", "A very long folder name", "Album"]
            .map(str::to_owned)
            .into();
        assert_eq!(fitted(&crumbs, 1000.0), crumbs);
        assert_eq!(fitted(&crumbs, 200.0), ["…", "Album"]);
        let kept = fitted(&crumbs, 320.0);
        assert_eq!(kept.first().map(String::as_str), Some("…"));
        assert_eq!(kept.last(), crumbs.last());
        assert_eq!(fitted(&crumbs, 10.0), ["…", "Album"]);
    }

    /// The page is the whole rows that fit under the tabs and the path
    /// line, above the panel's foot.
    #[test]
    fn the_page_is_the_rows_that_fit() {
        let head = 19.0;
        let over = top(head) + head / 2.0 + FOOT;
        assert_eq!(rows_in(over + 10.0 * height::ROW, head), 10);
        assert_eq!(rows_in(over + 10.0 * height::ROW - 1.0, head), 9);
        assert_eq!(rows_in(10.0, head), 0);
    }
}
