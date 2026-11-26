//! The settings screen: each row's label, its value between `‹ ›`, the
//! list of devices or of folders under them, and the notes at the bottom.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use jodel_playback::Headroom;

use super::colors::*;
use super::{bordered, keep_start, row_line};
use crate::core::{device_window, Gpu, SettingsRow, State};
use crate::view::layout::{SettingsLayout, SETTINGS_LABEL};

/// The settings screen as laid out in `s`.
pub(super) fn draw(frame: &mut Frame, state: &State, s: &SettingsLayout) {
    frame.render_widget(
        bordered(Line::from(Span::styled(
            " Settings ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))),
        s.pane,
    );
    for (i, (row, line_rect)) in SettingsRow::ALL.iter().zip(&s.lines).enumerate() {
        let selected =
            i == state.settings.row && !state.settings.picking && state.settings.folders.is_none();
        let label_style = if selected {
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(TEXT)
        };
        let marker = if selected { "▸ " } else { "  " };
        let indent = if row.under() { "  " } else { "" };
        let label = format!("{marker}{indent}{}", row.label());
        let label_rect = Rect::new(
            line_rect.rect.x,
            line_rect.rect.y,
            SETTINGS_LABEL.min(line_rect.rect.width),
            1,
        );
        frame.render_widget(Paragraph::new(Span::styled(label, label_style)), label_rect);
        let value = value_text(state, *row);
        let arrows = Style::new().fg(if selected { ACCENT } else { DIM });
        frame.render_widget(Paragraph::new(Span::styled("‹", arrows)), line_rect.less);
        frame.render_widget(Paragraph::new(Span::styled("›", arrows)), line_rect.more);
        let value_style = if selected {
            Style::new()
                .fg(TEXT)
                .bg(SELECTED)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(TEXT)
        };
        frame.render_widget(
            Paragraph::new(Span::styled(
                keep_start(&value, usize::from(line_rect.value.width)),
                value_style,
            )),
            line_rect.value,
        );
    }
    if let (Some(area), Some(pick)) = (s.devices, &state.settings.folders) {
        draw_folders(frame, state, pick, area);
    } else if let Some(area) = s.devices {
        draw_devices(frame, state, area);
    }
    let mut notes = Vec::new();
    notes.push(Line::from(Span::styled(
        match &state.config_file {
            Some(path) => format!("Saved to {}", path.display()),
            None => "Not saved: no config file".to_owned(),
        },
        Style::new().fg(DIM),
    )));
    if let Some(why) = state.data_move.as_ref().and_then(|m| m.failure.as_ref()) {
        notes.push(Line::from(Span::styled(why.clone(), Style::new().fg(FAIL))));
    }
    if state.null_output {
        notes.push(Line::from(Span::styled(
            "--null: this run plays into the null output, whatever device is set",
            Style::new().fg(WARN),
        )));
    }
    notes.push(Line::from(Span::styled(
        "↑↓ select   ←→ change   enter open   esc back",
        Style::new().fg(BORDER),
    )));
    frame.render_widget(Paragraph::new(notes), s.notes);
}

/// What `row` is set to, in words.
fn value_text(state: &State, row: SettingsRow) -> String {
    match row {
        SettingsRow::Device => format!("{}  ▾", state.device_label()),
        SettingsRow::Headroom => match state.config.headroom {
            Headroom::NONE => "none (lossy files -1.5 dB, for their overs)".to_owned(),
            headroom if headroom < jodel_playback::LOSSY_HEADROOM => {
                format!("{headroom} where the samples change, lossy files -1.5 dB")
            }
            headroom => format!("{headroom} where the samples change"),
        },
        SettingsRow::Tempo => if state.config.bpm {
            "on: tempo, beat pulse and beat analysis"
        } else {
            "off: no tempo, no beat pulse, nothing analysed"
        }
        .to_owned(),
        SettingsRow::Voting => if state.config.beat_voting {
            "on: the checked methods vote"
        } else {
            "off: one method decides"
        }
        .to_owned(),
        SettingsRow::Method(method) => {
            let c = &state.config;
            let mark = match (c.beat_voting, c.beat_methods.contains(&method)) {
                (true, true) => "[x]",
                (true, false) => "[ ]",
                (false, _) if c.beat_method == method => "(●)",
                (false, _) => "( )",
            };
            let what = match method {
                jodel_rhythm::Method::Flux => "onsets through two trackers",
                jodel_rhythm::Method::Network => "a neural network, our own inference",
            };
            format!("{mark} {what}")
        }
        SettingsRow::Stems => match &state.gpu {
            Gpu::Present(name) if state.config.stems => {
                format!("on: the cues listen to stems, separated on {name}")
            }
            Gpu::Present(_) => "off: the cues listen to the mix".to_owned(),
            Gpu::Absent => "unavailable: no GPU, the cues listen to the mix".to_owned(),
            Gpu::Looking => "looking for a GPU…".to_owned(),
        },
        SettingsRow::StemSwitch => if state.config.stems_on_bar {
            "on the next bar: stems go on and off on the beat"
        } else {
            "instantly: within a second or so"
        }
        .to_owned(),
        SettingsRow::SkipMix => if state.config.skip_mix {
            "mix in: beat-matched, by its transition below"
        } else {
            "cut: the next track starts at once"
        }
        .to_owned(),
        SettingsRow::SkipPreset => match state.config.skip_preset {
            None => format!(
                "the Transition tab's: {}",
                state.transition.draft.preset.name()
            ),
            Some(p) => format!("{} (its own)", p.name()),
        },
        SettingsRow::DataFolder => match (&state.data_move, &state.data_dir) {
            (Some(m), _) if m.failure.is_some() => "the move stopped: enter tries again".to_owned(),
            (Some(m), _) if !m.finished => format!(
                "{} {:.0} % moving to {}",
                bar_text(m.share, 10),
                m.share * 100.0,
                m.to.display()
            ),
            (_, Some(dir)) => format!("{}  ▾", dir.display()),
            (_, None) => "none  ▾".to_owned(),
        },
    }
}

/// A bar of `width` cells, `share` of it filled.
fn bar_text(share: f64, width: usize) -> String {
    let filled = ((share.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// The output devices to choose from, the chosen one marked.
fn draw_devices(frame: &mut Frame, state: &State, area: Rect) {
    let mut lines = Vec::new();
    match &state.settings.devices {
        None => lines.push(Line::from(Span::styled(
            "listing devices…",
            Style::new().fg(DIM),
        ))),
        Some(list) if list.is_empty() => lines.push(Line::from(Span::styled(
            "no output devices found",
            Style::new().fg(DIM),
        ))),
        Some(list) => {
            let start = device_window(state, usize::from(area.height));
            let w = usize::from(area.width);
            for (i, choice) in list.iter().enumerate().skip(start) {
                let chosen = choice.id == state.config.device;
                let mut l = row_line(
                    vec![
                        Span::styled(
                            if chosen { "● " } else { "○ " },
                            Style::new().fg(if chosen { GREEN } else { DIM }),
                        ),
                        Span::styled(choice.name.clone(), Style::new().fg(TEXT)),
                        Span::styled(format!("  {}", choice.summary), Style::new().fg(DIM)),
                    ],
                    Vec::new(),
                    w,
                );
                if i == state.settings.device_cursor {
                    l = l.style(Style::new().bg(SELECTED).add_modifier(Modifier::BOLD));
                }
                lines.push(l);
            }
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// The data folder's list: the choice of the folder shown, up, its
/// subfolders.
fn draw_folders(frame: &mut Frame, state: &State, pick: &crate::core::FolderPick, area: Rect) {
    use crate::core::FolderRow;
    let mut lines = Vec::new();
    let start = crate::core::folder_window(state, usize::from(area.height));
    let w = usize::from(area.width);
    let rows = pick.rows();
    for (i, row) in rows.iter().enumerate().skip(start) {
        let spans = match row {
            FolderRow::Choose(dir) => vec![
                Span::styled("✓ ", Style::new().fg(GREEN)),
                Span::styled(
                    format!(
                        "move the data folder here: {}",
                        dir.join(crate::dirs::DATA).display()
                    ),
                    Style::new().fg(TEXT),
                ),
            ],
            FolderRow::Up => vec![Span::styled("↑ ..", Style::new().fg(DIM))],
            FolderRow::Folder(dir) => vec![Span::styled(
                format!(
                    "▸ {}",
                    dir.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| dir.display().to_string())
                ),
                Style::new().fg(FOLDER),
            )],
        };
        let mut l = row_line(spans, Vec::new(), w);
        if i == pick.cursor {
            l = l.style(Style::new().bg(SELECTED).add_modifier(Modifier::BOLD));
        }
        lines.push(l);
    }
    if pick.folders.is_none() {
        lines.push(Line::from(Span::styled("listing…", Style::new().fg(DIM))));
    }
    frame.render_widget(Paragraph::new(lines), area);
}
