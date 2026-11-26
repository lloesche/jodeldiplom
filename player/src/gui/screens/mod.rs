//! What the window shows, screen by screen. A screen drawn by iced's own
//! widgets gets an arm here as it is ported; none has one yet. Every
//! screen is the old canvas filling the window ([`super::embed`]) with a
//! slot over it, the Mixer tab ([`super::mixer`]) or nothing: the same
//! shape, so the canvas's widget keeps its place in iced's tree. The
//! pointer over the canvas hides where the core says
//! ([`crate::core::karaoke::pointer::hidden`]); the view is built again
//! after every message, so it shows again as soon as it moves or the mode
//! ends.

use iced::widget::{opaque, pin, space, stack};
use iced::{Element, Widget as _};

use super::embed::Embed;
use super::mixer;
use super::window::Message;
use crate::core::layout::{self, Layout};
use crate::core::State;

/// The window's content for the player's `state` (`None`: no player yet,
/// or no longer).
pub(super) fn view<'a>(
    state: Option<&'a State>,
    embed: &Embed,
    caps: &'a mixer::Caps,
) -> Element<'a, Message> {
    let hide_pointer = state.is_some_and(crate::core::karaoke::pointer::hidden);
    let desk = state.and_then(|state| desk(state, embed, caps));
    stack![
        embed.view(hide_pointer),
        desk.unwrap_or_else(|| space().boxed())
    ]
    .boxed()
}

/// The Mixer tab, over its place on the old canvas, while it shows.
fn desk<'a>(
    state: &'a State,
    embed: &Embed,
    caps: &'a mixer::Caps,
) -> Option<Element<'a, Message>> {
    let Layout::Main(m) = layout::compute(state) else {
        return None;
    };
    let (Some(desk), None) = (&m.desk, &state.palette) else {
        return None;
    };
    let (cw, ch) = embed.cell();
    let r = m.rows;
    let view = mixer::view(state, caps, desk, r, (cw, ch));
    Some(
        pin(opaque(view))
            .x(f32::from(r.x) * cw)
            .y(f32::from(r.y) * ch)
            .boxed(),
    )
}
