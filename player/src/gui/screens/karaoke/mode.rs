//! Karaoke mode in the window's own widgets: the stage over the whole
//! window ([`StageView`]: the lyrics, the lane, the count-in, the score,
//! Next, the result; karaoke needs a GPU, so a window drawn without one
//! only says so: [`NO_STAGE`]); and its controls' bar over the bottom while
//! the pointer is near it ([`crate::core::karaoke::controls`]): the
//! transport, the time heard, the seek bar and the length, the voice
//! slider, the way out, and the keys under them. The bar is a translucent
//! raised band rising out of the stage in a soft gradient, fading in and
//! out as the core says ([`crate::core::karaoke::controls::handover`]);
//! the picture never moves for it.
//!
//! Every move of the pointer over the mode goes to the core (it shows the
//! pointer and, near the bottom, the controls); leaving the window, the
//! controls go. The pointer hides over all of it, controls too, while the
//! core says ([`crate::core::karaoke::pointer`]).

use std::f32::consts::PI;

use iced::gradient::Linear;
use iced::widget::{column, container, mouse_area, opaque, row, stack, text, Space};
use iced::{Alignment, Background, Color, Element, Fill, Padding, Widget as _};

use crate::core::karaoke::{self, controls};
use crate::core::keymap::Op;
use crate::core::sliders::Slider;
use crate::core::{Action, State};
use crate::gui::look::{self, size, space};
use crate::gui::pointer::{Grab, WheelOn};
use crate::gui::stage_view::StageView;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::bottom::Seek;

/// The controls' bar's height in points: what the stage keeps the line
/// under a result clear above
/// ([`crate::view::karaoke::KaraokePicture::bar`]).
pub const BAR: f32 = 96.0;
/// How far its gradient rises over the stage above it, in points.
const RISE: f32 = 56.0;
/// Two lines of text, in points: how much further up than the bar the
/// pointer brings the controls ([`reach`]).
const TWO_LINES: f32 = 38.0;
/// The width the times take beside the seek bar, in points.
const TIME: f32 = 52.0;
/// The voice slider's track, in points.
const SLIDER: f32 = 160.0;
/// How opaque the bar's band is, under its controls.
const BAND: f32 = 0.86;

/// The share of a window `height` points high the controls' bar takes,
/// from its bottom.
pub fn bar_share(height: f32) -> f32 {
    (BAR / height.max(1.0)).min(1.0)
}

/// The share of a window `height` points high, from its bottom, where the
/// pointer brings the controls up: the bar and two lines
/// ([`crate::core::Fit::controls_reach`]).
pub fn reach(height: f32) -> f32 {
    ((BAR + TWO_LINES) / height.max(1.0)).min(1.0)
}

/// What the mode says in a window drawn without a GPU (iced's CPU
/// renderer), where the stage cannot draw: karaoke needs one.
pub const NO_STAGE: &str =
    "Karaoke needs a graphics card, and this window is drawn without one. esc leaves.";

/// Karaoke mode for `state`: the stage (in a window drawn without a GPU,
/// [`NO_STAGE`] alone), the controls' bar over its bottom while they show,
/// the pointer heard everywhere over it.
pub fn view<'a>(state: &'a State, stage: &StageView) -> Element<'a, Message> {
    let size = stage.size();
    let picture: Element<'a, Message> = if stage.gpu() {
        stage.view()
    } else {
        container(
            text(NO_STAGE)
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT_DIM),
        )
        .center(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            ..container::Style::default()
        })
        .boxed()
    };
    let bar: Element<'a, Message> = if controls::shown(state) {
        let shown = controls::handover(controls::fade(state)).1 as f32;
        column![Space::new().height(Fill), self::bar(state, shown)].boxed()
    } else {
        Space::new().boxed()
    };
    let height = size.height.max(1.0);
    let heard = mouse_area(stack![picture, bar])
        .on_move(move |at| {
            Message::Act(Action::PointerMoved {
                from_bottom: ((height - at.y) / height).clamp(0.0, 1.0),
            })
        })
        .on_exit(Message::Act(Action::PointerLeft))
        .boxed();
    widgets::hide_pointer(heard, karaoke::pointer::hidden(state))
}

/// The controls' bar, `shown` of the way in (0 to 1): its gradient rising
/// out of the stage, the band under its controls.
fn bar<'a>(state: &State, shown: f32) -> Element<'a, Message> {
    let top = RISE / (RISE + BAR);
    // Eased in over the rise, so it has no edge (iced's gradients take
    // eight stops: seven over the rise, the last at the bottom).
    let band = |t: f32| {
        let eased = t * t * (3.0 - 2.0 * t);
        look::alpha(look::RAISED, BAND * eased * shown)
    };
    let gradient = (0..7u8).fold(Linear::new(PI), |g, i| {
        let t = f32::from(i) / 6.0;
        g.add_stop(t * top, band(t))
    });
    let gradient = gradient.add_stop(1.0, band(1.0));
    let body = container(column![controls_row(state, shown), hint(shown)].spacing(space::S))
        .padding(Padding::from([space::M, space::XL]))
        .height(BAR)
        .width(Fill)
        .align_y(Alignment::Center);
    container(column![Space::new().height(RISE), opaque(body)])
        .width(Fill)
        .style(move |_| container::Style {
            background: Some(Background::Gradient(gradient.into())),
            ..container::Style::default()
        })
        .boxed()
}

/// The transport, the time heard, the seek bar and the length, the voice
/// slider and the way out, `shown` of the way in.
fn controls_row<'a>(state: &State, shown: f32) -> Element<'a, Message> {
    let seek = Seek::of(state);
    let tone = |c: Color| look::alpha(c, c.a * shown);
    let time = |s: &str, color: Color| {
        text(s.to_owned())
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(tone(color))
    };
    let times = if seek.track {
        look::TEXT_DIM
    } else {
        look::TEXT_FAINT
    };
    let mut track = widgets::track(seek.share)
        .held(seek.held)
        .opacity(shown)
        .id("karaoke-seek");
    if seek.track && !karaoke::seek_refused(state) {
        track = track.on_press(|bounds, at| Message::Grab(Grab::Seek(bounds), at));
    }
    row![
        transport(state, shown),
        time(&seek.elapsed, times)
            .width(TIME)
            .align_x(Alignment::End),
        Element::from(track),
        time(&seek.total, times).width(TIME),
        voice(state, shown),
        Element::from(
            widgets::ghost("Leave")
                .leading(Icon::Cross)
                .opacity(shown)
                .id("karaoke-leave")
                .on_press(Message::Act(Action::Op(Op::Back)))
        ),
    ]
    .spacing(space::M)
    .align_y(Alignment::Center)
    .boxed()
}

/// Previous and next (each holds that song for START) round play or
/// pause: pause while a song is heard; otherwise play, greyed, for START
/// is the stage's (the controls have none).
fn transport<'a>(state: &State, shown: f32) -> Element<'a, Message> {
    let act = |op| Message::Act(Action::Op(op));
    let play = if karaoke::pausable(state) {
        widgets::round(Icon::Pause)
            .id("pause")
            .on_press(act(Op::PlayPause))
    } else {
        widgets::round(Icon::Play).id("play")
    };
    row![
        Element::from(
            widgets::button::icon(Icon::Previous)
                .opacity(shown)
                .id("previous")
                .on_press(act(Op::Previous))
        ),
        Element::from(play.opacity(shown)),
        Element::from(
            widgets::button::icon(Icon::Next)
                .opacity(shown)
                .id("next")
                .on_press(act(Op::Next))
        ),
    ]
    .spacing(space::XS)
    .align_y(Alignment::Center)
    .boxed()
}

/// The voice slider: its label, its track (a press grabs it, the wheel
/// steps it) and its level.
fn voice<'a>(state: &State, shown: f32) -> Element<'a, Message> {
    let slider = Slider::Voice;
    let tone = |c: Color| look::alpha(c, c.a * shown);
    let (label, value) = (look::TEXT_DIM, look::TEXT);
    let track = widgets::track(slider.share(state))
        .held(state.karaoke.dragging == Some(slider))
        .opacity(shown)
        .width(SLIDER)
        .id("karaoke-voice")
        .on_press(move |bounds, at| Message::Grab(Grab::Slider(slider, bounds), at));
    let parts = row![
        text(slider.label())
            .size(size::LABEL)
            .font(look::font::label())
            .color(tone(label)),
        Element::from(track),
        text(slider.text(state).trim().to_owned())
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(tone(value))
            .width(TIME),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center);
    mouse_area(parts)
        .on_scroll(move |delta| {
            let step = |by| Action::KaraokeStep { slider, by };
            Message::Steps(
                WheelOn::Slider(slider),
                delta,
                Box::new((step(1), step(-1))),
            )
        })
        .boxed()
}

/// The keys, in a line under the controls.
fn hint<'a>(shown: f32) -> Element<'a, Message> {
    text(crate::view::karaoke::hint())
        .size(size::CAPTION)
        .font(look::font::body())
        .color(look::alpha(look::TEXT_FAINT, shown))
        .wrapping(text::Wrapping::None)
        .ellipsis(text::Ellipsis::End)
        .width(Fill)
        .align_x(Alignment::Center)
        .boxed()
}
