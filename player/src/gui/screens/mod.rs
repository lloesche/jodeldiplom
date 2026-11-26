//! What the window shows, screen by screen. A screen drawn by iced's own
//! widgets gets an arm here as it is ported: the setup screen
//! ([`setup`]), Now playing ([`now`]), and the settings ([`settings`])
//! and the keys ([`help`]) over the bottom bar ([`native`]), the command
//! palette over them. Every other screen is the old canvas filling the
//! window ([`super::embed`]), the covers and the lanes over it where it
//! left their cells ([`super::widgets::cover`], [`super::lanes`], neither
//! taking the pointer), and slots over them: the Mixer tab
//! ([`super::mixer`]) or nothing, the bottom bar ([`bottom`]) or nothing,
//! the list pane with its tabs ([`list`]) or nothing, and the command
//! palette over all of it, the lanes too ([`palette`]) or nothing; the
//! same shape, so each widget keeps its place in iced's tree (the tabs'
//! pill slides from one side to the next). The pointer over the canvas
//! hides where the core says
//! ([`crate::core::karaoke::pointer::hidden`]); the view is built again
//! after every message, so it shows again as soon as it moves or the mode
//! ends.

use iced::widget::{column, container, opaque, pin, space, stack};
use iced::{Element, Fill, Size, Widget as _};

use super::embed::Embed;
use super::lanes::Lanes;
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
pub mod settings;
pub mod setup;

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
    let palette = state.and_then(palette::view);
    stack![
        embed.view(hide_pointer),
        covers.view(),
        lanes.view(),
        desk.unwrap_or_else(|| space().boxed()),
        bar.unwrap_or_else(|| space().boxed()),
        list.unwrap_or_else(|| space().boxed()),
        palette.unwrap_or_else(|| space().boxed())
    ]
    .boxed()
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
