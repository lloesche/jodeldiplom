//! What the window shows, screen by screen, in iced's widgets: the setup
//! screen ([`setup`]), Now playing ([`now`]), the lyrics editor
//! ([`lyrics`], its lane over it), the settings ([`settings`]) and the keys
//! ([`help`]) over the bottom bar
//! ([`bottom`]), and the main screen laid out in rows and columns from
//! the window's measures ([`crate::gui::metrics`]): the tabs' header
//! along the top ([`list::tabs`]), under it the side pane (the browser
//! or the queue, [`list`]; the Transition tab, [`transition`]; the
//! Karaoke tab, [`karaoke`]; the Mixer tab, [`super::mixer`]) with the
//! right column beside it (the cover and the text under it, [`right`]) or
//! the track's summary under it, the Karaoke tab's voice lane under it,
//! and the bottom bar along the foot. The lanes are drawn in their part
//! of the screen ([`super::lanes`], taking no pointer), the covers in
//! their places ([`super::widgets::cover`]); the command palette goes
//! over everything ([`palette`]), the Karaoke tab's sync's panel over the
//! main screen. Karaoke mode is its own over the whole window
//! ([`karaoke::mode`], on its stage, [`super::stage_view`]).
//!
//! The tabs' header is the main screen's first part on every side, so
//! the tabs keep their place in iced's tree (the pill slides from one
//! side to the next).

use iced::widget::{column, container, mouse_area, opaque, row, space, stack, Space};
use iced::{mouse, Alignment, Background, Element, Fill, Point, Size, Widget as _};

use super::lanes::{LaneSpot, LaneSpots, Metrics, Part, Rows};
use super::look;
use super::metrics::{self, Main};
use super::mixer;
use super::stage_view::StageView;
use super::widgets;
use super::widgets::cover::{self, Covers};
use super::window::Message;
use crate::core::keymap::Op;
use crate::core::{Action, Screen, Side, State};
use crate::spectrum::SpectrumFrame;
use crate::view::karaoke::Tab;

pub mod bottom;
pub mod help;
pub mod karaoke;
pub mod list;
pub mod lyrics;
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

/// What the window's screens draw with beside the player's state.
pub(super) struct Parts<'a> {
    /// Karaoke mode's stage.
    pub stage: &'a StageView,
    /// The Mixer tab's faders' caps.
    pub caps: &'a mixer::Caps,
    /// The covers, fitted to their places ([`covers`]).
    pub covers: &'a Covers,
    /// The lanes, framed where they go ([`lane_spots`]).
    pub lanes: &'a super::lanes::Lanes,
    /// The spectrum heard, for the bottom bar.
    pub heard: &'a SpectrumFrame,
    /// The window's size, in points, and its pixels a point.
    pub size: Size,
    pub scale: f32,
}

/// The window's content for what is `shown`.
pub(super) fn view<'a>(shown: Shown<'a>, parts: &Parts<'a>) -> Element<'a, Message> {
    let state = match shown {
        Shown::Setup(status) => return setup::view(&status),
        Shown::Player(Some(state)) => state,
        Shown::Player(None) => return ground(space().boxed()),
    };
    let screen = match state.screen {
        Screen::NowPlaying => now::view(state, parts.covers),
        Screen::Settings => page(state, settings::view(state), parts),
        Screen::Help => page(state, help::view(state), parts),
        Screen::Lyrics => lyrics::view(state, parts),
        Screen::Main => main(state, parts),
        Screen::Karaoke => karaoke::mode::view(state, parts.stage),
    };
    let palette = palette::view(state).unwrap_or_else(|| space().boxed());
    stack![screen, palette].boxed()
}

/// `content` on the window's ground.
fn ground(content: Element<'_, Message>) -> Element<'_, Message> {
    container(content)
        .width(Fill)
        .height(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            ..container::Style::default()
        })
        .boxed()
}

/// A screen the window lays out whole (the settings, the keys), over the
/// bottom bar as the main screen has it, without the spectrum.
pub(super) fn page<'a>(
    state: &State,
    page: Element<'a, Message>,
    parts: &Parts<'a>,
) -> Element<'a, Message> {
    let bar = bottom::view(state, parts.heard, parts.size.width, 0.0);
    column![
        container(page).width(Fill).height(Fill),
        container(opaque(bar))
            .width(Fill)
            .height(metrics::EDGE + metrics::BAND)
    ]
    .boxed()
}

/// The main screen: the tabs' header, the side pane with the right column
/// beside it or the summary under it, the Karaoke tab's voice lane, the
/// bottom bar; the sync's panel over it while one shows.
fn main<'a>(state: &'a State, parts: &Parts<'a>) -> Element<'a, Message> {
    let m = metrics::main(parts.size, state.side);
    let header = container(list::tabs(state, m.pane.width))
        .padding(iced::Padding::from([0.0, metrics::MARGIN]))
        .width(Fill)
        .height(metrics::HEADER)
        .align_y(Alignment::Center);
    let tab = (state.side == Side::Karaoke).then(|| Tab::of(state));
    let pane = container(pane(state, &m, tab.as_ref(), parts))
        .width(Fill)
        .height(Fill)
        .id("pane");
    let mut body = row![pane.boxed()].spacing(metrics::GAP);
    if let Some(r) = m.right {
        body = body.push(
            container(right_column(state, &m, parts.covers))
                .width(r.width)
                .height(Fill)
                .id("right")
                .boxed(),
        );
    }
    let gap = || Space::new().height(metrics::MARGIN).boxed();
    let mut screen = column![
        header.boxed(),
        container(body)
            .padding(iced::Padding::from([0.0, metrics::MARGIN]))
            .height(Fill)
            .boxed(),
        gap()
    ];
    if let Some(s) = m.summary {
        screen = screen.push(
            container(right::summary(state))
                .padding(iced::Padding::from([0.0, metrics::MARGIN]))
                .width(Fill)
                .height(s.height)
                .boxed(),
        );
        screen = screen.push(gap());
    }
    if let Some(v) = m.voice {
        let lane = voice(tab.as_ref(), parts.lanes, v.position());
        screen = screen.push(
            container(container(lane).width(Fill).height(Fill).id("voice"))
                .padding(iced::Padding::from([0.0, metrics::MARGIN]))
                .width(Fill)
                .height(v.height)
                .boxed(),
        );
        screen = screen.push(gap());
    }
    let bar = bottom::view(state, parts.heard, m.bar.width, m.spectrum);
    screen = screen.push(
        container(opaque(bar))
            .width(Fill)
            .height(m.bar.height)
            .boxed(),
    );
    let sync = tab
        .as_ref()
        .and_then(|t| t.sync.as_ref().map(karaoke::tab::sync))
        .unwrap_or_else(|| space().boxed());
    // Where no lanes show, their widget still lets go of what the GPU
    // holds of them.
    let shown = match state.side {
        Side::Transition => true,
        Side::Karaoke => m.voice.is_some() && tab.as_ref().is_some_and(|t| t.voice.is_none()),
        Side::Browser | Side::Queue | Side::Mixer => false,
    };
    let lanes = if shown {
        space().boxed()
    } else {
        parts.lanes.view(Point::ORIGIN)
    };
    stack![ground(screen.boxed()), lanes, sync].boxed()
}

/// The side pane for `state` as `m` lays it out.
fn pane<'a>(
    state: &'a State,
    m: &Main,
    tab: Option<&Tab>,
    parts: &Parts<'a>,
) -> Element<'a, Message> {
    let size = m.pane.size();
    match (state.side, tab) {
        (Side::Transition, _) => {
            let room = transition::Room {
                pane: m.pane,
                scale: parts.scale,
            };
            stack![
                transition::view(state, &room),
                parts.lanes.view(m.pane.position())
            ]
            .boxed()
        }
        (Side::Karaoke, Some(tab)) => {
            karaoke::tab::view(tab, list::turn(state), size, parts.covers.art(0))
        }
        (Side::Mixer, _) => desk(state, parts.caps, size),
        _ => list::view(state, size.width),
    }
}

/// The line of the Mixer tab's keys.
const DESK_FOOT: f32 = 28.0;

/// The Mixer tab in a pane of `size`: the desk, its keys under it.
fn desk<'a>(state: &'a State, caps: &'a mixer::Caps, size: Size) -> Element<'a, Message> {
    let desk = Size::new(size.width, (size.height - DESK_FOOT).max(0.0));
    column![
        mixer::view(state, caps, desk),
        container(widgets::hints(&mixer::keys()))
            .padding(iced::Padding::from([0.0, look::space::S]))
            .height(DESK_FOOT)
            .align_y(Alignment::Center)
    ]
    .boxed()
}

/// The right column: the cover at its top (its click opens Now playing;
/// a card in its place without art), the track's text under it. The
/// wheel over it scrolls the list.
fn right_column<'a>(state: &State, m: &Main, covers: &'a Covers) -> Element<'a, Message> {
    let width = m.right.map_or(0.0, |r| r.width);
    let mut parts = iced::widget::Column::<Element<'a, Message>>::new();
    if let Some(place) = m.cover {
        let side = place.width.min(place.height);
        let shown = covers
            .art(0)
            .unwrap_or_else(|| karaoke::tab::placeholder(side, side));
        let cover = mouse_area(shown)
            .on_press(Message::Act(Action::Op(Op::NowPlaying)))
            .interaction(mouse::Interaction::Pointer);
        parts = parts
            .push(
                container(cover)
                    .width(place.width)
                    .height(place.height)
                    .boxed(),
            )
            .push(Space::new().height(metrics::UNDER_COVER).boxed());
    }
    parts = parts.push(right::view(state, width));
    mouse_area(container(parts).width(Fill).height(Fill).clip(true))
        .on_scroll(Message::Wheel)
        .boxed()
}

/// The Karaoke tab's voice lane, its top left at `origin` in the window:
/// the lane, or what it says while it has nothing to show.
fn voice<'a>(
    tab: Option<&Tab>,
    lanes: &'a super::lanes::Lanes,
    origin: Point,
) -> Element<'a, Message> {
    match tab.and_then(|t| t.voice) {
        Some(note) => karaoke::tab::voice(note),
        None => lanes.view(origin),
    }
}

/// The places of the covers `state`'s screen shows in a window of `size`,
/// in pixels at `scale` pixels a point, in the order the screen takes
/// them ([`Covers::art`]).
pub(super) fn covers(state: &State, size: Size, scale: f32) -> Vec<(u32, u32)> {
    let place = match state.screen {
        Screen::NowPlaying => Some(now::place(size)),
        Screen::Main => {
            let m = metrics::main(size, state.side);
            match state.side {
                Side::Karaoke => Some(karaoke::tab::cover(m.pane.size()).size()),
                Side::Browser | Side::Queue => m.cover.map(|c| c.size()),
                Side::Transition | Side::Mixer => None,
            }
        }
        Screen::Settings | Screen::Help | Screen::Karaoke | Screen::Lyrics => None,
    };
    place
        .filter(|p| p.width > 0.0 && p.height > 0.0)
        .map(|p| cover::pixels(p, scale))
        .into_iter()
        .collect()
}

/// The lanes `state`'s screen shows in a window of `size` at `scale`
/// pixels a point, painted at `metrics`, `heard` seconds into the playing
/// track: the Transition tab's, the Karaoke tab's voice lane, the lyrics
/// editor's.
pub fn lane_spots(
    state: &State,
    size: Size,
    scale: f32,
    metrics: Metrics,
    heard: f64,
) -> LaneSpots {
    let mut spots = Vec::new();
    if let Some(room) = transition::Room::of(state, size, scale) {
        spots = transition::spots(state, &room, heard);
    }
    if let Some(room) = lyrics::Room::of(state, size, scale) {
        spots = lyrics::spots(state, &room);
    }
    let m = metrics::main(size, state.side);
    if let (Screen::Main, Some(v)) = (state.screen, m.voice) {
        let px = |points: f32| (points * scale).round().max(0.0) as usize;
        let rect = (
            px(v.x),
            px(v.y),
            px(v.x + v.width) - px(v.x),
            px(v.y + v.height) - px(v.y),
        );
        let (cw, ch) = (metrics.cell.0.max(1), metrics.cell.1.max(1));
        let letters = (rect.2 / cw).min(usize::from(u16::MAX)) as u16;
        let rows = (rect.3 / ch).min(usize::from(u16::MAX)) as u16;
        if let Some(plot) = crate::view::karaoke::voice_plot(state, letters, rows, heard) {
            let rows = Rows {
                above: 0,
                full: rect.3,
            };
            spots.push(LaneSpot {
                rect,
                part: Part::Wave(Box::new(plot), rows),
            });
        }
    }
    LaneSpots { metrics, spots }
}
