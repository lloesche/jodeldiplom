//! The bottom bar: the spectrum with its labels, the time heard and the
//! seek bar, the buttons, and the message line.

use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::*;
use super::plot::{self, Plot};
use super::{keep_start, Pictures, Scene};
use crate::core::layout::Bottom;
use crate::core::text::clock;
use crate::core::{Level, Screen, State};

/// The time heard (right-aligned in `elapsed`), the seek bar filling in the
/// signal path's colour with its head, and the length; a drag shows where
/// it would go.
pub(super) fn draw_progress(
    frame: &mut Frame,
    state: &State,
    elapsed: Rect,
    bar: Rect,
    total: Rect,
) {
    let duration = state.playing.duration();
    let position = match (state.seek_drag, duration) {
        (Some(f), Some(d)) => f * d,
        _ => state.playing.position.max(0.0),
    };
    let has_track = state.playing.track.is_some();
    let time_style = Style::new().fg(if has_track { TEXT } else { BORDER });
    frame.render_widget(
        Paragraph::new(Span::styled(clock(position), time_style))
            .alignment(HorizontalAlignment::Right),
        elapsed,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            duration.map_or_else(|| "--:--".to_owned(), clock),
            time_style,
        )),
        total,
    );
    let fraction = match duration {
        Some(d) if d > 0.0 => (position / d).clamp(0.0, 1.0),
        _ => 0.0,
    };
    let filled = (fraction * f64::from(bar.width)).round() as u16;
    let color = ACCENT;
    let buf = frame.buffer_mut();
    for i in 0..bar.width {
        let (symbol, style) = if i < filled {
            ("━", Style::new().fg(color))
        } else {
            ("─", Style::new().fg(TRACK))
        };
        buf[(bar.x + i, bar.y)].set_symbol(symbol).set_style(style);
    }
    if has_track && bar.width > 0 {
        let head = filled.min(bar.width - 1);
        buf[(bar.x + head, bar.y)]
            .set_symbol("●")
            .set_style(Style::new().fg(color));
    }
}

/// The bottom bar as laid out in `b`.
pub(super) fn draw(frame: &mut Frame, scene: &Scene, b: &Bottom, pictures: &mut dyn Pictures) {
    let state = scene.state;
    if let Some(area) = b.spectrum {
        let plot = Plot::new(area, b.labels, scene.spectrum, plot::zones(state));
        pictures.spectrum(frame, &plot);
        plot.draw_labels(frame.buffer_mut());
    }
    draw_progress(frame, state, b.elapsed, b.bar, b.total);
    let buf = frame.buffer_mut();
    for button in &b.buttons {
        let off = button
            .op
            .stem()
            .is_some_and(|stem| crate::core::stems::is_off(state, stem));
        let style = if off {
            Style::new()
                .fg(DIM)
                .bg(BUTTON)
                .add_modifier(Modifier::CROSSED_OUT)
        } else {
            Style::new()
                .fg(TEXT)
                .bg(BUTTON)
                .add_modifier(Modifier::BOLD)
        };
        buf.set_string(button.rect.x, button.rect.y, &button.label, style);
    }
    let message = match (&state.message, crate::core::preview::line(state)) {
        (Some(m), _) => Span::styled(
            m.text.clone(),
            Style::new().fg(if m.level == Level::Error { FAIL } else { TEXT }),
        ),
        // A preview says so while it lasts (a message from then shows over it).
        (None, Some(line)) => Span::styled(line, Style::new().fg(TEXT)),
        (None, None) => {
            let mut text = match state.playing.state {
                jodel_playback::PlayState::Playing => "playing".to_owned(),
                jodel_playback::PlayState::Paused => "paused".to_owned(),
                jodel_playback::PlayState::Stopped => "stopped".to_owned(),
            };
            if state.screen == Screen::Main {
                text.push_str("  ·  : commands  / filter  ? keys");
            }
            Span::styled(text, Style::new().fg(DIM))
        }
    };
    let text = keep_start(&message.content, usize::from(b.message.width));
    frame.render_widget(
        Paragraph::new(Span::styled(text, message.style)).alignment(HorizontalAlignment::Center),
        b.message,
    );
}
