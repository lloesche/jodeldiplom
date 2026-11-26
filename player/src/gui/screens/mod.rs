//! What the window shows, screen by screen. A screen drawn by iced's own
//! widgets gets an arm here as it is ported: the setup screen
//! ([`setup`]), Now playing ([`now`]), and the settings ([`settings`])
//! and the keys ([`help`]) over the bottom bar ([`native`]), the command
//! palette over them. Every other screen is the old canvas filling the
//! window ([`super::embed`]), the covers over it where it left their
//! cells ([`super::widgets::cover`], not taking the pointer), the
//! Transition tab ([`transition`]) or nothing, the lanes over all of it
//! ([`super::lanes`], not taking the pointer), and slots over them: the
//! Mixer tab
//! ([`super::mixer`]) or nothing, the bottom bar ([`bottom`]) or nothing,
//! the list pane with its tabs ([`list`]) or nothing, the right column's
//! text under the cover or the summary under the list ([`right`]) or
//! nothing, and the command palette over all of it, the lanes too
//! ([`palette`]) or nothing; the same shape, so each widget keeps its
//! place in iced's tree (the tabs' pill slides from one side to the
//! next). The pointer over the canvas hides where the core says
//! ([`crate::core::karaoke::pointer::hidden`]); the view is built again
//! after every message, so it shows again as soon as it moves or the mode
//! ends.

use iced::widget::{column, container, opaque, pin, space, stack};
use iced::{Element, Fill, Point, Rectangle, Size, Widget as _};

use super::embed::Embed;
use super::lanes::Lanes;
use super::look;
use super::mixer;
use super::widgets::cover::Covers;
use super::window::Message;
use crate::core::{Screen, Side, State};
use crate::spectrum::SpectrumFrame;
use crate::view::layout::{self, Layout, BAR_ROWS};

pub mod bottom;
pub mod help;
pub mod list;
pub mod now;
pub mod palette;
pub mod right;
pub mod settings;
pub mod setup;
pub mod transition;

/// What the window has to show.
pub(super) enum Shown<'a> {
    /// The setup screen, where setup stands.
    Setup(crate::setup::Status),
    /// The player's state (`None`: no player yet, or no longer).
    Player(Option<&'a State>),
}

/// The window's content for what is `shown`, the spectrum `heard` in its
/// bottom bar.
pub(super) fn view<'a>(
    shown: Shown<'a>,
    embed: &Embed,
    caps: &'a mixer::Caps,
    covers: &'a Covers,
    lanes: &'a Lanes,
    heard: &SpectrumFrame,
) -> Element<'a, Message> {
    let state = match shown {
        Shown::Setup(status) => return setup::view(&status),
        Shown::Player(state) => state,
    };
    if let Some(screen) = state.and_then(|state| native(state, embed, covers, heard)) {
        let palette = state.and_then(palette::view);
        return stack![screen, palette.unwrap_or_else(|| space().boxed())].boxed();
    }
    let hide_pointer = state.is_some_and(crate::core::karaoke::pointer::hidden);
    let desk = state.and_then(|state| desk(state, embed, caps));
    let bar = state.and_then(|state| bar(state, embed, heard));
    let list = state.and_then(|state| list(state, embed));
    let right = state.and_then(|state| right(state, embed, covers));
    let palette = state.and_then(palette::view);
    let transition = state.and_then(|state| transition(state, embed));
    stack![
        embed.view(hide_pointer),
        covers.view(),
        transition.unwrap_or_else(|| space().boxed()),
        lanes.view(),
        desk.unwrap_or_else(|| space().boxed()),
        bar.unwrap_or_else(|| space().boxed()),
        list.unwrap_or_else(|| space().boxed()),
        right.unwrap_or_else(|| space().boxed()),
        palette.unwrap_or_else(|| space().boxed())
    ]
    .boxed()
}

/// The right column's text under the cover, its edges in line with the
/// cover's (with no art, the placeholder's; with art much narrower than
/// the column, two thirds of it), from a gap under it to the column's
/// foot; or the summary over its place under the list where the
/// window is too narrow for the column. On the main screen's browser and
/// queue sides.
fn right<'a>(state: &State, embed: &Embed, covers: &Covers) -> Option<Element<'a, Message>> {
    let Layout::Main(m) = layout::compute(state, embed.grid()) else {
        return None;
    };
    let (cw, ch) = embed.cell();
    let cells = |x: u16, y: u16, w: u16, h: u16| {
        Rectangle::new(
            Point::new(f32::from(x) * cw, f32::from(y) * ch),
            Size::new(f32::from(w) * cw, f32::from(h) * ch),
        )
    };
    let pinned = |content: Element<'a, Message>, at: Rectangle| {
        pin(opaque(
            container(content)
                .width(at.width)
                .height(at.height)
                .clip(true),
        ))
        .x(at.x)
        .y(at.y)
        .boxed()
    };
    if let Some(mini) = m.mini {
        let at = cells(mini.x, mini.y, mini.width, mini.height);
        return Some(pinned(right::summary(state), at));
    }
    let (column, info) = (m.right?, m.info?);
    let column = cells(column.x, column.y, column.width, column.height);
    // The placeholder's edge runs through the middle of its border cells.
    let placeholder = Rectangle {
        x: column.x + cw / 2.0,
        width: column.width - cw,
        ..column
    };
    // The text starts a gap under the art as drawn (fitted, it can end
    // above its cells), or under the placeholder's bottom edge.
    let (x, width, top) = match (m.cover, covers.first()) {
        (Some(_), Some(art)) => (
            art.x,
            art.width.max(placeholder.width * 2.0 / 3.0),
            art.y + art.height + look::space::L,
        ),
        (Some(c), None) => (
            placeholder.x,
            placeholder.width,
            (f32::from(c.y + c.height) - 0.5) * ch + look::space::L,
        ),
        (None, _) => (
            placeholder.x,
            placeholder.width,
            f32::from(info.y) * ch + ch / 2.0,
        ),
    };
    // The place goes on to the column's right edge, so the wheel scrolls
    // the list beside the text too.
    let at = Rectangle::new(
        Point::new(x, top),
        Size::new(
            column.x + column.width - x,
            (column.y + column.height - top).max(0.0),
        ),
    );
    Some(pinned(right::view(state, width), at))
}

/// The list pane over its place on the old canvas, on the main screen:
/// whole on the browser's and the queue's side, its tabs alone over a tab
/// of its own's top border.
fn list<'a>(state: &State, embed: &Embed) -> Option<Element<'a, Message>> {
    let Layout::Main(m) = layout::compute(state, embed.grid()) else {
        return None;
    };
    let (cw, ch) = embed.cell();
    let rows = if state.side.wide() { 1 } else { m.list.height };
    let room = list::Room {
        width: f32::from(m.list.width) * cw,
        height: f32::from(rows) * ch,
        cell: (cw, ch),
    };
    Some(
        pin(opaque(
            iced::widget::container(list::view(state, room))
                .width(room.width)
                .height(room.height),
        ))
        .x(f32::from(m.list.x) * cw)
        .y(f32::from(m.list.y) * ch)
        .boxed(),
    )
}

/// The screen shown where it is one iced lays out whole: Now playing; the
/// settings and the keys, each over the bottom bar where the main screen
/// has it (its last rows of cells), the same bar.
fn native<'a>(
    state: &'a State,
    embed: &Embed,
    covers: &'a Covers,
    heard: &SpectrumFrame,
) -> Option<Element<'a, Message>> {
    let page = match state.screen {
        Screen::NowPlaying => return Some(now::view(state, covers)),
        Screen::Settings => settings::view(state),
        Screen::Help => help::view(state),
        Screen::Main | Screen::Karaoke => return None,
    };
    let grid = embed.grid();
    let (cw, ch) = embed.cell();
    let top = f32::from(grid.rows.saturating_sub(BAR_ROWS)) * ch;
    let bar = bottom::view(state, heard, f32::from(grid.cols) * cw, 0.0);
    Some(column![container(page).width(Fill).height(top), opaque(bar)].boxed())
}

/// The Transition tab over the list pane's place on the old canvas, while
/// it shows: under the lanes, which it leaves room for (they take no
/// pointer: it takes it over them).
fn transition<'a>(state: &State, embed: &Embed) -> Option<Element<'a, Message>> {
    let room = transition::Room::of(state, embed.grid(), embed.cell(), embed.scale())?;
    Some(transition::view(state, &room))
}

/// The bottom bar, over its place on the old canvas (to the window's
/// right and bottom edges), on the screens that have one.
fn bar<'a>(state: &State, embed: &Embed, heard: &SpectrumFrame) -> Option<Element<'a, Message>> {
    let grid = embed.grid();
    let place = *layout::compute(state, grid).bottom()?;
    let (cw, ch) = embed.cell();
    let width = f32::from(grid.cols) * cw;
    let spectrum = f32::from(place.spectrum) * ch;
    let view = bottom::view(state, heard, width, spectrum);
    Some(pin(opaque(view)).y(f32::from(place.area.y) * ch).boxed())
}

/// The Mixer tab, pinned over its place on the old canvas (the list
/// pane's inside) while it shows.
fn desk<'a>(
    state: &'a State,
    embed: &Embed,
    caps: &'a mixer::Caps,
) -> Option<Element<'a, Message>> {
    if state.side != Side::Mixer || state.palette.is_some() {
        return None;
    }
    let Layout::Main(m) = layout::compute(state, embed.grid()) else {
        return None;
    };
    let (cw, ch) = embed.cell();
    let r = m.rows;
    let size = Size::new(f32::from(r.width) * cw, f32::from(r.height) * ch);
    Some(
        pin(opaque(mixer::view(state, caps, size)))
            .x(f32::from(r.x) * cw)
            .y(f32::from(r.y) * ch)
            .boxed(),
    )
}
