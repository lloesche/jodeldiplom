//! What the window shows, screen by screen. A screen drawn by iced's own
//! widgets gets an arm here as it is ported: the setup screen
//! ([`setup`]). Every other screen is the old canvas filling the window
//! ([`super::embed`]), the covers over it where it left their cells
//! ([`super::widgets::cover`], not taking the pointer), and slots over
//! them: the Mixer tab ([`super::mixer`]) or nothing, and the bottom bar
//! ([`bottom`]) or nothing; the same shape, so each widget keeps its place
//! in iced's tree. The pointer over the canvas hides where the core says
//! ([`crate::core::karaoke::pointer::hidden`]); the view is built again
//! after every message, so it shows again as soon as it moves or the mode
//! ends.

use iced::widget::{opaque, pin, space, stack};
use iced::{Element, Size, Widget as _};

use super::embed::Embed;
use super::mixer;
use super::widgets::cover::Covers;
use super::window::Message;
use crate::core::{Side, State};
use crate::spectrum::SpectrumFrame;
use crate::view::layout::{self, Layout};

pub mod bottom;
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
    heard: &SpectrumFrame,
) -> Element<'a, Message> {
    let state = match shown {
        Shown::Setup(status) => return setup::view(&status),
        Shown::Player(state) => state,
    };
    let hide_pointer = state.is_some_and(crate::core::karaoke::pointer::hidden);
    let desk = state.and_then(|state| desk(state, embed, caps));
    let bar = state.and_then(|state| bar(state, embed, heard));
    stack![
        embed.view(hide_pointer),
        covers.view(),
        desk.unwrap_or_else(|| space().boxed()),
        bar.unwrap_or_else(|| space().boxed())
    ]
    .boxed()
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
