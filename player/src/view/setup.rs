//! The setup screen ([`crate::setup`]): what the player gets before it can
//! be used, step by step, with a bar for the step under way and one for
//! the whole; on a failure, what went wrong, what to do, and two buttons:
//! Try again and Quit (there is no skipping: without its files the player
//! is not what it should be).

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Widget, Wrap};
use ratatui::Frame;

use super::colors::*;
use crate::setup::{Status, StepState, Unit};

/// Where the buttons were drawn, for the mouse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buttons {
    /// Try again (after a failure).
    pub retry: Option<Rect>,
    /// Quit.
    pub quit: Option<Rect>,
}

impl Buttons {
    /// The button at `(col, row)`.
    pub fn at(&self, col: u16, row: u16) -> Option<Choice> {
        let inside = |r: Option<Rect>| {
            r.is_some_and(|r| {
                col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
            })
        };
        if inside(self.retry) {
            Some(Choice::Retry)
        } else if inside(self.quit) {
            Some(Choice::Quit)
        } else {
            None
        }
    }
}

/// What the person chose on the setup screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Try again, after a failure.
    Retry,
    /// Quit the player.
    Quit,
}

/// What a key means on the setup screen: `Enter` or `r` try again (after a
/// failure), `q` or `Esc` quit. `key` is the key's name as the window gives
/// it: one character, `"enter"` or `"esc"`.
pub fn choice(key: &str, failed: bool) -> Option<Choice> {
    match key {
        "enter" | "r" | "R" if failed => Some(Choice::Retry),
        "q" | "Q" | "esc" => Some(Choice::Quit),
        _ => None,
    }
}

fn size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
}

/// A bar `width` cells wide, `share` of it filled.
fn bar(share: f64, width: u16, fill: Color) -> Line<'static> {
    let width = usize::from(width);
    let filled = ((share.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    Line::from(vec![
        Span::styled("█".repeat(filled), Style::new().fg(fill)),
        Span::styled("█".repeat(width - filled), Style::new().fg(TRACK)),
    ])
}

/// What the step under way has done: "812 MB of 1.88 GB · 48 MB/s · about
/// 23 s left", "tensor 120 of 401".
fn detail(status: &Status) -> String {
    match status.unit {
        Some(Unit::Bytes) if status.total > 0 => {
            let mut text = format!("{} of {}", size(status.done), size(status.total));
            if let Some(rate) = status.rate.filter(|r| *r > 0.0) {
                let left = (status.total.saturating_sub(status.done)) as f64 / rate;
                text.push_str(&format!(" · {:.0} MB/s", rate / 1e6));
                if left >= 1.0 {
                    let left = left.round() as u64;
                    text.push_str(&if left >= 90 {
                        format!(" · about {} min left", left.div_ceil(60))
                    } else {
                        format!(" · about {left} s left")
                    });
                }
            }
            text
        }
        Some(Unit::Tensors) if status.total > 0 => {
            format!("tensor {} of {}", status.done, status.total)
        }
        Some(Unit::Tensors) => "reading the download…".to_owned(),
        _ => String::new(),
    }
}

/// Draws the setup screen over the whole frame; where its buttons are.
pub fn draw(frame: &mut Frame, status: &Status) -> Buttons {
    let area = frame.area();
    frame.render_widget(Clear, area);
    frame.render_widget(Block::new().style(Style::new().bg(BG)), area);
    let width = area.width.saturating_sub(4).clamp(20, 84);
    let inner = width.saturating_sub(4);
    let text = |s: String| Line::from(Span::styled(s, Style::new().fg(TEXT)));
    let dim = |s: String| Line::from(Span::styled(s, Style::new().fg(DIM)));
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            "Getting the player ready",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
        dim(
            "This happens once. The player needs these files and checks each against a \
             checksum built into it. Once they are here it never needs the network for them \
             again."
                .to_owned(),
        ),
        Line::default(),
    ];
    for (i, step) in status.steps.iter().enumerate() {
        let failed_here = status.failure.is_some() && status.current == Some(i);
        let (mark, color) = match step.state {
            StepState::Done => ("✓", GREEN),
            StepState::Working if failed_here => ("✗", FAIL),
            StepState::Working => ("▶", ACCENT),
            StepState::ToDo => ("·", DIM),
        };
        let style = match step.state {
            StepState::ToDo => Style::new().fg(DIM),
            _ => Style::new().fg(TEXT),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{mark} "), Style::new().fg(color)),
            Span::styled(step.title.clone(), style),
        ]));
    }
    if status.steps.is_empty() && status.finished.is_none() && status.failure.is_none() {
        lines.push(dim("Looking for the files…".to_owned()));
    }
    lines.push(Line::default());
    if let Some(step) = status.current.and_then(|i| status.steps.get(i)) {
        if status.failure.is_none() {
            lines.push(text(step.title.clone()));
            lines.push(dim(step.why.clone()));
            lines.push(bar(status.step_share(), inner, ACCENT));
            lines.push(dim(detail(status)));
            lines.push(Line::default());
        }
    }
    lines.push(dim(format!("All of it: {:.0} %", status.overall() * 100.0)));
    lines.push(bar(status.overall(), inner, GREEN));
    let mut buttons = Buttons::default();
    let button_row;
    if let Some(failure) = &status.failure {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            failure.what.clone(),
            Style::new().fg(FAIL),
        )));
        lines.push(text(failure.advice.clone()));
        lines.push(Line::default());
        button_row = Some(lines.len());
        lines.push(Line::default());
    } else {
        lines.push(Line::default());
        lines.push(dim(
            "q quits (what is downloaded whole is kept for the next start)".to_owned(),
        ));
        button_row = None;
    }
    // The height the text takes wrapped, then the box centred.
    let wrapped: u16 = lines
        .iter()
        .map(|l| {
            let w = l.width() as u16;
            w.div_ceil(inner.max(1)).max(1)
        })
        .sum();
    let height = (wrapped + 4).min(area.height);
    let rect = Rect::new(
        area.x + (area.width.saturating_sub(width + 4)) / 2 + 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .title(Span::styled(" Jodeldiplom ", Style::new().fg(DIM)))
        .style(Style::new().bg(BG));
    let body = block.inner(rect);
    let body = Rect::new(
        body.x + 1,
        body.y + 1,
        body.width.saturating_sub(2),
        body.height.saturating_sub(1),
    );
    frame.render_widget(block, rect);
    Paragraph::new(lines.clone())
        .wrap(Wrap { trim: false })
        .render(body, frame.buffer_mut());
    if let Some(row) = button_row {
        // The wrapped rows before the buttons' row.
        let before: u16 = lines[..row]
            .iter()
            .map(|l| (l.width() as u16).div_ceil(inner.max(1)).max(1))
            .sum();
        let y = body.y + before;
        if y < body.y + body.height {
            let draw = |frame: &mut Frame, x: u16, label: &str, primary: bool| {
                let w = label.chars().count() as u16 + 4;
                let r = Rect::new(x, y, w.min(body.x + body.width - x), 1);
                let style = if primary {
                    Style::new().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::new().fg(TEXT).bg(BUTTON)
                };
                Paragraph::new(Span::styled(format!("  {label}  "), style))
                    .render(r, frame.buffer_mut());
                r
            };
            let retry = draw(frame, body.x, "Try again (Enter)", true);
            let quit = draw(frame, retry.x + retry.width + 2, "Quit (Q)", false);
            buttons = Buttons {
                retry: Some(retry),
                quit: Some(quit),
            };
        }
    }
    buttons
}
