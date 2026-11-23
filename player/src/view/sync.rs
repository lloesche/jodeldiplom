//! The sync on the Karaoke tab ([`crate::core::sync`]): its row (the
//! button, and where the delay in force comes from), and its panel while
//! one shows: what to do, the clicks as dots lighting up as they play (in
//! the accent colour once answered), how many were heard, and at the end
//! the delay measured, with its buttons.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Widget};

use super::colors::*;
use super::keep_start;
use crate::core::layout::{KaraokeTabLayout, SLIDER_LABEL};
use crate::core::sync::{Outcome, CLICKS};
use crate::core::text::signed_ms;
use crate::core::State;

/// Where the delay in force comes from as the tab says it, with the
/// singing's correction and the nudge on it where there are: `from your
/// singing`, `measured · your singing +30 ms · nudged −20 ms`.
pub fn source_text(state: &State) -> String {
    let lag = &state.karaoke.lag;
    let corrected = (lag.corrected * 1000.0).round() as i32;
    let corrected = if corrected == 0 {
        String::new()
    } else {
        format!(" · your singing {}", signed_ms(corrected))
    };
    let nudged = match lag.nudge {
        Some(n) => format!(" · nudged {}", signed_ms((n * 1000.0).round() as i32)),
        None => String::new(),
    };
    format!("{}{corrected}{nudged}", lag.source.name())
}

/// The keys while a sync shows, for the pane's bottom border: keep only
/// once there is a delay to keep, again once it has ended.
pub fn footer(state: &State) -> &'static str {
    match state.karaoke.sync.as_ref().and_then(|s| s.outcome.as_ref()) {
        None => "say “ta” with each click · esc cancel",
        Some(Outcome::Measured(_)) => "enter keep · r again · esc cancel",
        Some(_) => "r again · esc cancel",
    }
}

/// The sync's row: its name and key, its button, where the delay comes
/// from.
pub fn row(buf: &mut Buffer, state: &State, layout: &KaraokeTabLayout) {
    let r = layout.sync;
    if r.width == 0 {
        return;
    }
    buf.set_string(r.x, r.y, "Sync  (s)", Style::new().fg(TEXT));
    let button = r.x + SLIDER_LABEL;
    let lit = Style::new().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD);
    buf.set_string(
        button,
        r.y,
        keep_start(" sync ", usize::from(r.right().saturating_sub(button))),
        lit,
    );
    let at = r.right() + 2;
    let room = usize::from(layout.title.right().saturating_sub(at));
    if room > 4 {
        buf.set_string(
            at,
            r.y,
            keep_start(&source_text(state), room),
            Style::new().fg(DIM),
        );
    }
}

/// The sync's panel in `area`, over the tab, and its `buttons`.
pub fn panel(
    buf: &mut Buffer,
    state: &State,
    area: Rect,
    buttons: &[(Rect, crate::core::keymap::Op)],
) {
    let Some(session) = &state.karaoke.sync else {
        return;
    };
    if area.width < 20 || area.height < 5 {
        return;
    }
    Clear.render(area, buf);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(ACCENT))
        .style(Style::new().bg(BG))
        .title(Span::styled(
            " Sync: how late your voice comes ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    block.render(area, buf);
    let text = Style::new().fg(TEXT);
    let dim = Style::new().fg(DIM);
    let mut lines = vec![
        Line::default(),
        Line::from(Span::styled(
            " Say “ta” (or clap) with each click.",
            text.add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            " The microphone hears you: this is the delay the scoring sees.",
            dim,
        )),
        Line::default(),
    ];
    // The clicks: played, answered.
    let mut dots = vec![Span::raw("  ")];
    for k in 0..CLICKS {
        let (symbol, style) = if k >= session.played {
            ("○", dim)
        } else if session.answers.get(k).copied().flatten().is_some() {
            ("●", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
        } else {
            ("●", text)
        };
        dots.push(Span::styled(format!("{symbol}  "), style));
    }
    dots.push(Span::styled(
        format!("  {} of {} heard", session.heard(), session.played),
        dim,
    ));
    lines.push(Line::from(dots));
    lines.push(Line::default());
    lines.push(match &session.outcome {
        None => Line::default(),
        Some(Outcome::Measured(d)) => Line::from(vec![
            Span::styled(" Your delay: ", text),
            Span::styled(
                format!("{} ms", (d * 1000.0).round() as i32),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Some(Outcome::TooFew(n)) => Line::from(Span::styled(
            format!(" Didn't hear enough ({n} of {CLICKS} together): again?"),
            Style::new().fg(WARN),
        )),
        Some(Outcome::Failed(why)) => Line::from(Span::styled(
            format!(" The clicks stopped: {why}"),
            Style::new().fg(FAIL),
        )),
    });
    let w = usize::from(inner.width);
    let lines: Vec<Line> = lines
        .into_iter()
        .map(|l| {
            if l.width() > w {
                let s: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
                Line::from(Span::styled(keep_start(&s, w), l.spans[0].style))
            } else {
                l
            }
        })
        .collect();
    Paragraph::new(lines).render(inner, buf);
    // The buttons: keep lit, the others plain.
    for &(rect, op) in buttons {
        let label = session
            .buttons()
            .into_iter()
            .find(|b| b.1 == op)
            .map_or("", |b| b.0);
        let style = if op == crate::core::keymap::Op::SyncKeep {
            Style::new().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(TEXT).bg(BUTTON)
        };
        buf.set_string(rect.x, rect.y, format!(" {label} "), style);
    }
}
