//! Karaoke mode in the window's own widgets: the stage over the whole
//! window ([`StageView`]: the lyrics, the lane, the count-in, the score,
//! START, the result), or without a GPU the mode's lines as text over the
//! cover's blur ([`fallback`]); and its controls' bar over the bottom while
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
use iced::widget::text::{Rich, Span};
use iced::widget::{
    column, container, image, mouse_area, opaque, row, span, stack, text, Column, Space,
};
use iced::{Alignment, Background, Color, Element, Fill, Length, Padding, Size, Widget as _};

use crate::core::karaoke::{self, controls};
use crate::core::keymap::Op;
use crate::core::sliders::Slider;
use crate::core::{Action, Gpu, State};
use crate::gui::look::{self, size, space};
use crate::gui::pointer::{Grab, WheelOn};
use crate::gui::stage_view::StageView;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::bottom::Seek;
use crate::view::karaoke::{line_color, points_text, KaraokePicture, LineKind};

/// The controls' bar's height in points: what the stage keeps START clear
/// above ([`KaraokePicture::bar`]).
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

/// Karaoke mode for `state`: the stage (or, without a GPU, the lines over
/// the cover's blur), the controls' bar over its bottom while they show,
/// the pointer heard everywhere over it.
pub fn view<'a>(state: &'a State, stage: &StageView) -> Element<'a, Message> {
    let size = stage.size();
    let picture: Element<'a, Message> = if stage.gpu() {
        stage.view()
    } else {
        fallback(state, stage.ground(), size)
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
/// steps it) and its level; faint where the voice cannot be turned down
/// (no GPU to separate it).
fn voice<'a>(state: &State, shown: f32) -> Element<'a, Message> {
    let slider = Slider::Voice;
    let live = matches!(state.gpu, Gpu::Present(_));
    let tone = |c: Color| look::alpha(c, c.a * shown);
    let (label, value) = if live {
        (look::TEXT_DIM, look::TEXT)
    } else {
        (look::TEXT_FAINT, look::TEXT_FAINT)
    };
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

/// Karaoke mode without a GPU, in a window of `size` points: the mode's
/// lines in the middle over the cover's blur (`ground`), the line being
/// sung filling in the accent as it is sung, the count-in large at the
/// left, the score top right; and while the mode waits for START a small
/// "enter: START" under the lines (a result's own lines end with START).
pub fn fallback<'a>(
    state: &State,
    ground: Option<image::Handle>,
    size: Size,
) -> Element<'a, Message> {
    let picture = crate::view::karaoke::picture(state, bar_share(size.height), state.heard());
    let big = (size.height / 11.0).clamp(20.0, 120.0).round();
    let ground: Element<'a, Message> = match ground {
        Some(handle) => image(handle)
            .width(Fill)
            .height(Fill)
            .content_fit(iced::ContentFit::Fill)
            .filter_method(image::FilterMethod::Linear)
            .boxed(),
        None => container(Space::new())
            .width(Fill)
            .height(Fill)
            .style(|_| container::Style {
                background: Some(Background::Color(look::BG)),
                ..container::Style::default()
            })
            .boxed(),
    };
    let lines = container(lines(&picture, big))
        .width(Fill)
        .height(Fill)
        .center_x(Fill)
        .center_y(Fill)
        .padding(Padding::from([0.0, space::XXL]));
    let count: Element<'a, Message> =
        match picture.count.or(picture.cue.map(|c| (c.number, c.through))) {
            Some((n, through)) => container(
                text(n.to_string())
                    .size(big * 2.4)
                    .font(look::font::title())
                    .color(look::alpha(
                        rgb(state.config.karaoke_theme.accent()),
                        1.0 - (through as f32).powi(3),
                    )),
            )
            .width(Fill)
            .height(Fill)
            .center_y(Fill)
            .padding(Padding::from([0.0, size.width * 0.1]))
            .boxed(),
            None => Space::new().boxed(),
        };
    let score: Element<'a, Message> = match picture.score {
        Some(score) => {
            let points = text(points_text(score.points))
                .size(big * 0.4)
                .font(look::font::numbers())
                .color(look::ACCENT);
            let mut figures = Column::new().push(points.boxed()).align_x(Alignment::End);
            if let Some(rung) = score.rung_label() {
                figures = figures.push(
                    text(rung)
                        .size(big * 0.25)
                        .font(look::font::label())
                        .color(look::TEXT_DIM)
                        .boxed(),
                );
            }
            container(figures)
                .width(Fill)
                .align_x(Alignment::End)
                .padding(space::XL)
                .boxed()
        }
        None => Space::new().boxed(),
    };
    stack![ground, lines, count, score].boxed()
}

/// `picture`'s lines, stacked and centred, the line sung `big` points
/// high and the rest smaller; "enter: START" under them while the mode
/// waits for it (not under a result: its lines end with START).
fn lines<'a>(picture: &KaraokePicture, big: f32) -> Element<'a, Message> {
    let mut lines = Column::new().spacing(big * 0.45).align_x(Alignment::Center);
    for line in &picture.lines {
        let share = match line.kind {
            LineKind::Current => 1.0,
            LineKind::Next | LineKind::Previous => 0.62,
            LineKind::Info => 0.42,
            LineKind::Warning => 0.36,
        };
        let faded = if line.kind == LineKind::Previous {
            0.5
        } else {
            1.0
        };
        let alpha = faded * line.alpha.clamp(0.0, 1.0) as f32;
        let rest = look::alpha(rgb(line_color(line.kind)), alpha);
        // The line sung fills in the accent as it is sung.
        let chars = line.text.chars().count();
        let sung = if line.kind == LineKind::Current {
            (chars as f64 * line.fill.clamp(0.0, 1.0)).round() as usize
        } else {
            0
        };
        let at = line
            .text
            .char_indices()
            .nth(sung)
            .map_or(line.text.len(), |(i, _)| i);
        let (done, ahead) = line.text.split_at(at);
        let spans: Vec<Span<'a, ()>> = [(done, look::alpha(look::ACCENT, alpha)), (ahead, rest)]
            .into_iter()
            .filter(|(part, _)| !part.is_empty())
            .map(|(part, color)| span(part.to_owned()).color(color))
            .collect();
        let rich: Rich<'a, (), Message> = Rich::with_spans(spans);
        lines = lines.push(
            rich.size((big * share).round())
                .font(look::font::title())
                .align_x(Alignment::Center)
                .boxed(),
        );
    }
    if picture.start.is_some() && picture.result.is_none() {
        lines = lines.push(
            text("enter: START")
                .size((big * 0.42).round())
                .font(look::font::title())
                .color(rgb(line_color(LineKind::Info)))
                .boxed(),
        );
    }
    lines.width(Length::Shrink).boxed()
}

/// A colour of the window's (sRGB bytes) as iced takes it.
fn rgb([r, g, b]: [u8; 3]) -> Color {
    Color::from_rgb8(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::Screen;

    /// Without a GPU the mode's lines are text in the middle: while the
    /// mode waits for START, "enter: START" under them says it (no pill to
    /// click); a count-in's number large beside them, and then no START.
    #[test]
    fn without_a_gpu_the_lines_say_start_in_words() {
        let mut state = State::new("/m".into(), Config::default());
        state.screen = Screen::Karaoke;
        state.now_ms = 10_000;
        let size = Size::new(1280.0, 800.0);
        let shows = |state: &State, words: &str| {
            // A renderer of its own: a turn at the GPU's setup, as every
            // device the tests make takes.
            let _turn = jodel_gpu::setting_up();
            let mut sim = iced_test::simulator::Simulator::with_size(
                crate::gui::iced_settings(),
                size,
                fallback(state, None, size),
            );
            sim.find(words).is_ok()
        };
        assert!(shows(&state, "Queue some songs, then START"));
        assert!(shows(&state, "enter: START"));
        // A paused song going on: its count-in, START not asked for.
        state.karaoke.resume_at = Some(state.now_ms + 1_500);
        assert!(shows(&state, "2"));
        assert!(!shows(&state, "enter: START"));
    }
}
