//! The command palette over the list pane's rows: what is typed, the
//! commands it matches with their keys.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;

use super::colors::*;
use super::row_line;
use crate::core::{palette_window, State};

/// The palette in `area`, where it is open.
pub(super) fn draw(frame: &mut Frame, state: &State, area: Rect) {
    let Some(palette) = &state.palette else {
        return;
    };
    let (matches, start) = palette_window(state, usize::from(area.height));
    let mut lines = vec![Line::from(vec![
        Span::styled(": ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(palette.text.clone(), Style::new().fg(TEXT)),
        Span::styled("▏", Style::new().fg(ACCENT)),
    ])];
    let w = usize::from(area.width);
    for (i, entry) in matches.iter().enumerate().skip(start) {
        let mut l = row_line(
            vec![Span::styled(
                format!("  {}", entry.name),
                Style::new().fg(TEXT),
            )],
            vec![Span::styled(entry.keys.clone(), Style::new().fg(DIM))],
            w,
        );
        if i == palette.cursor {
            l = l.style(Style::new().bg(SELECTED).add_modifier(Modifier::BOLD));
        }
        lines.push(l);
    }
    if matches.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no command matches",
            Style::new().fg(DIM),
        )));
    }
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines), area);
}
