//! The help screen: every key, by context, as its search leaves them.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::*;
use super::{bordered, search_line};
use crate::core::{help_shown, State};
use crate::view::layout::HelpLayout;

/// The help screen's lines, as its search leaves them (the same count as
/// [`crate::core::help_line_count`]).
fn help_lines(state: &State) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for section in help_shown(state) {
        lines.push(Line::from(Span::styled(
            section.title,
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        for (keys, help) in section.lines {
            lines.push(Line::from(vec![
                Span::styled(format!("  {keys:<22}"), Style::new().fg(TEXT)),
                Span::styled(help, Style::new().fg(DIM)),
            ]));
        }
        lines.push(Line::default());
    }
    lines
}

/// The help screen as laid out in `h`, scrolled as far as the state says.
pub(super) fn draw(frame: &mut Frame, state: &State, h: &HelpLayout) {
    frame.render_widget(
        bordered(Line::from(Span::styled(
            " Keys ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )))
        .title_bottom(Line::from(Span::styled(
            " ↑↓ scroll   / search   esc back ",
            Style::new().fg(DIM),
        ))),
        h.pane,
    );
    let lines: Vec<Line> = help_lines(state)
        .into_iter()
        .skip(state.help_scroll)
        .collect();
    if let (Some(rect), Some(search)) = (h.search, &state.help_search) {
        // As the browser's filter line: what is typed, and how many keys
        // it leaves.
        let shown: usize = help_shown(state).iter().map(|s| s.lines.len()).sum();
        search_line(frame, rect, &search.text, search.editing, shown);
        if shown == 0 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    format!("No key or text has “{}”", search.text),
                    Style::new().fg(DIM),
                )),
                h.text,
            );
            return;
        }
    }
    frame.render_widget(Paragraph::new(lines), h.text);
}
