//! The list pane: its box with the Browser / Queue toggle and its footer,
//! and what it shows (the browser's and the queue's rows, the command
//! palette over them, the Transition and Karaoke tabs; the Mixer tab is the
//! window's own), the filter line under it.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use super::colors::*;
use super::{bordered, command_palette, karaoke, row_line, search_line, sync, transition};
use super::{keep_start, Pictures, Scene};
use crate::browser::EntryKind;
use crate::core::layout::MainLayout;
use crate::core::text::{self, clock, width};
use crate::core::{browser_rows, list_cursor, row_entry, Row, Side, State};

/// The Mixer tab's bottom border: its keys.
const DESK_FOOTER: &str = "← → strip · ↑ ↓ fader · home 0 dB · m mute · s solo · e echo";

/// The list pane of the main screen, and the filter line under it.
pub(super) fn draw_pane(
    frame: &mut Frame,
    scene: &Scene,
    m: &MainLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    // The list pane with the Browser / Queue toggle in its border.
    let mut title = vec![Span::raw("")];
    for (label, side, shown) in text::toggle(state) {
        let style = match (side, shown) {
            (None, _) => Style::new().fg(BORDER),
            (Some(_), true) => Style::new()
                .fg(Color::Rgb(20, 20, 24))
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
            (Some(_), false) => Style::new().fg(DIM),
        };
        title.push(Span::styled(label, style));
    }
    let footer = match state.side {
        Side::Browser => {
            let b = &state.browser;
            let mut text = if crate::browser::is_drives(&b.dir) {
                "Drives".to_owned()
            } else {
                b.dir.to_string_lossy().into_owned()
            };
            if let Some(search) = &b.search {
                let status = if search.done { "" } else { ", searching…" };
                text = format!("{} found below {text}{status}", search.hits.len());
            } else if b.loading {
                text.push_str(&format!(" (listing {})", text::spinner(state)));
            }
            keep_end(&text, usize::from(m.list.width.saturating_sub(4)))
        }
        Side::Queue => match state.queue.current {
            Some(i) => format!("playing {} of {}", i + 1, state.queue.paths.len()),
            None => format!("{} in the queue", state.queue.paths.len()),
        },
        // The zoom first: cut, the keys go.
        Side::Transition => keep_start(
            &transition::footer(state),
            usize::from(
                m.list.width.saturating_sub(4).saturating_sub(
                    m.transition
                        .as_ref()
                        .and_then(|x| x.fit)
                        .map_or(0, |f| f.width + 1),
                ),
            ),
        ),
        Side::Karaoke if state.karaoke.sync.is_some() => sync::footer(state).into(),
        Side::Karaoke => "enter Karaoke mode · ↑ ↓ row · ← → its value · s sync".into(),
        Side::Mixer => DESK_FOOTER.into(),
    };
    let block = bordered(Line::from(title)).title_bottom(Line::from(Span::styled(
        format!(" {footer} "),
        Style::new().fg(DIM),
    )));
    frame.render_widget(block, m.list);
    if state.palette.is_some() {
        command_palette::draw(frame, state, m.rows);
    } else {
        match state.side {
            Side::Browser => draw_browser(frame, state, m.rows),
            Side::Queue => draw_queue(frame, state, m.rows),
            Side::Transition => {
                if let Some(layout) = &m.transition {
                    transition::draw(frame, scene, layout, pictures);
                    if let Some(fit) = layout.fit {
                        frame.buffer_mut().set_string(
                            fit.x,
                            fit.y,
                            crate::core::layout::FIT_LYRICS,
                            Style::new().fg(TEXT).bg(BORDER),
                        );
                    }
                }
            }
            Side::Karaoke => {
                if let Some(layout) = &m.karaoke {
                    karaoke::draw_tab(frame, scene, layout, pictures);
                }
            }
            // The window draws the desk in iced's widgets over its rows
            // ([`crate::gui::mixer`]).
            Side::Mixer => {}
        }
    }
    if let (Some(rect), Some(filter)) = (m.filter, &state.browser.filter) {
        let shown = browser_rows(state)
            .iter()
            .filter(|r| !matches!(r, Row::SearchBelow))
            .count();
        search_line(frame, rect, &filter.text, filter.editing, shown);
    }
}

/// Shortens `text` to `max` cells, keeping its end (for paths).
fn keep_end(text: &str, max: usize) -> String {
    if width(text) <= max {
        return text.to_owned();
    }
    let mut out: Vec<char> = Vec::new();
    let mut used = 1;
    for c in text.chars().rev() {
        let w = width(c.encode_utf8(&mut [0; 4]));
        if used + w > max {
            break;
        }
        used += w;
        out.push(c);
    }
    out.reverse();
    format!("…{}", out.into_iter().collect::<String>())
}

/// A queue row's icons: each grey until the track has that thing, lit once
/// it has, in the order the work finds them, so they light up one after
/// the other as the queue is got ready:
/// - `♯` its key (found as soon as the file is decoded);
/// - `▥` its stems (separated alongside, on the GPU);
/// - `♩` its tempo (decided last, from the beats and the stems);
/// - `¶` its lyrics (looked up and timed alongside, by another worker);
///   orange when no trusted source had them and machine learning heard
///   them from the voice.
fn status_spans(state: &State, path: &std::path::Path) -> Vec<Span<'static>> {
    use crate::core::karaoke::SongLyrics;
    use crate::core::{Job, LyricsFacts, QueueFacts, Rhythm, RhythmFacts};
    let mut facts = state.facts.get(path).cloned().unwrap_or_default();
    // Known here already (the songs wanted now), though not told yet.
    if let Some(Rhythm::Ready(r)) = state.rhythm.get(path) {
        if facts.tempo.is_none() {
            facts.analysed(&RhythmFacts::of(&r.analysis));
        }
    }
    let QueueFacts {
        key,
        stems,
        tempo,
        lyrics,
        cover,
    } = facts;
    // Heard by machine learning (no trusted source): lit in orange.
    let known = state.karaoke.lyrics.get(path);
    let heard = lyrics == Some(LyricsFacts::Heard) || known.is_some_and(SongLyrics::heard);
    let lyrics = heard
        || matches!(lyrics, Some(LyricsFacts::Timed | LyricsFacts::Untimed))
        || matches!(known, Some(SongLyrics::Found { .. }));
    // Each job's icons, lit when it found them (the lyrics heard by
    // machine learning in orange): the analysis's key, stems and tempo, the
    // lyrics, the cover taken apart; for the jobs the queue is got ready
    // by, in that order.
    let icons = |job: &Job| -> Vec<(&'static str, bool, bool)> {
        match job {
            Job::Rhythm => vec![
                ("♯", key == Some(true), false),
                ("▥", stems == Some(true), false),
                ("♩", tempo == Some(true), false),
            ],
            Job::Lyrics => vec![("¶", lyrics, heard)],
            Job::Cover => vec![("◐", cover == Some(true), false)],
        }
    };
    let mut out = Vec::new();
    for job in [Job::Rhythm, Job::Lyrics, Job::Cover] {
        if !state.preparing.contains(&job) {
            continue;
        }
        for (icon, has, warn) in icons(&job) {
            let color = match (has, warn) {
                (true, true) => WARN,
                (true, false) => ACCENT,
                (false, _) => DIM,
            };
            out.push(Span::styled(icon, Style::new().fg(color)));
            out.push(Span::raw(" "));
        }
    }
    out.push(Span::raw(" "));
    out
}

/// A file's chip (its format) right-aligned in a column, and its length.
fn chip_spans(state: &State, path: &std::path::Path) -> Vec<Span<'static>> {
    match state.details.get(path) {
        Some(d) => {
            let chip_style = if d.error.is_some() {
                Style::new().fg(FAIL)
            } else {
                Style::new().fg(DIM)
            };
            // Right-aligned in a column, so what stands left of it lines up.
            let mut out = vec![Span::styled(format!("{:>9}", d.chip), chip_style)];
            if let Some(seconds) = d.seconds {
                out.push(Span::styled(
                    format!(" {:>6}", clock(seconds)),
                    Style::new().fg(DIM),
                ));
            }
            out
        }
        None => vec![Span::styled(format!("{:>9}", "…"), Style::new().fg(BORDER))],
    }
}

fn draw_browser(frame: &mut Frame, state: &State, area: Rect) {
    let b = &state.browser;
    let rows = browser_rows(state);
    // A note under the rows (the `..` row stays usable above it).
    let only_parent = rows
        .iter()
        .all(|&r| row_entry(state, r).is_some_and(|e| e.kind == EntryKind::Parent));
    let note = if let Some(error) = &b.error {
        Some(Span::styled(error.clone(), Style::new().fg(FAIL)))
    } else if b.loading {
        Some(Span::styled(
            format!("{} listing…", text::spinner(state)),
            Style::new().fg(DIM),
        ))
    } else if b.search.is_some() && rows.is_empty() {
        Some(Span::styled("nothing found yet", Style::new().fg(DIM)))
    } else if only_parent && b.search.is_none() {
        Some(Span::styled(
            "no folders or music files here",
            Style::new().fg(DIM),
        ))
    } else {
        None
    };
    let shown = u16::try_from(rows.len())
        .unwrap_or(u16::MAX)
        .min(area.height);
    let note_area = (shown < area.height).then(|| Rect {
        y: area.y + shown,
        height: area.height - shown,
        ..area
    });
    if let (Some(note), Some(at)) = (note, note_area) {
        frame.render_widget(Paragraph::new(note), at);
    }
    if rows.is_empty() {
        return;
    }
    let (cursor, scroll) = list_cursor(state);
    let w = usize::from(area.width);
    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .skip(scroll)
        .take(usize::from(area.height))
        .map(|(i, &row)| {
            let mut l = match (row, row_entry(state, row)) {
                (Row::SearchBelow, _) => {
                    let query = b.filter.as_ref().map_or("", |f| f.text.as_str());
                    row_line(
                        vec![Span::styled(
                            format!("» search all subfolders for “{query}”  (enter)"),
                            Style::new().fg(ACCENT),
                        )],
                        Vec::new(),
                        w,
                    )
                }
                (_, Some(e)) => match e.kind {
                    EntryKind::Parent => row_line(
                        vec![Span::styled("  ..", Style::new().fg(DIM))],
                        vec![Span::styled("up ", Style::new().fg(BORDER))],
                        w,
                    ),
                    EntryKind::Dir => row_line(
                        vec![
                            Span::styled("▸ ", Style::new().fg(FOLDER)),
                            Span::styled(format!("{}/", e.label()), Style::new().fg(FOLDER)),
                        ],
                        Vec::new(),
                        w,
                    ),
                    EntryKind::Playlist => row_line(
                        vec![
                            Span::styled("≡ ", Style::new().fg(FOLDER)),
                            Span::styled(e.label().to_owned(), Style::new().fg(TEXT)),
                        ],
                        vec![Span::styled("playlist ", Style::new().fg(BORDER))],
                        w,
                    ),
                    EntryKind::File => {
                        let details = state.details.get(&e.path);
                        let title = if e.relative.is_some() {
                            e.label().to_owned()
                        } else {
                            details
                                .and_then(|d| d.title.clone())
                                .unwrap_or_else(|| e.name.clone())
                        };
                        let current = state
                            .playing
                            .track
                            .as_ref()
                            .is_some_and(|t| t.info.path == e.path);
                        let mark = if current { "▶ " } else { "♪ " };
                        let mark_style = if current {
                            Style::new().fg(ACCENT)
                        } else {
                            Style::new().fg(BORDER)
                        };
                        row_line(
                            vec![
                                Span::styled(mark, mark_style),
                                Span::styled(title, Style::new().fg(TEXT)),
                            ],
                            chip_spans(state, &e.path),
                            w,
                        )
                    }
                },
                _ => Line::default(),
            };
            if i == cursor && state.side == Side::Browser {
                l = l.style(Style::new().bg(SELECTED).add_modifier(Modifier::BOLD));
            }
            l
        })
        .collect();
    let rows_area = Rect {
        height: shown,
        ..area
    };
    frame.render_widget(Paragraph::new(lines), rows_area);
}

fn draw_queue(frame: &mut Frame, state: &State, area: Rect) {
    let q = &state.queue;
    if q.paths.is_empty() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("The queue is empty.", Style::new().fg(DIM))),
                Line::from(Span::styled(
                    "enter queues a file or playlist, shift+enter plays it too; alt+enter plays a file with its whole folder.",
                    Style::new().fg(DIM),
                )),
            ])
            .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    let (cursor, scroll) = list_cursor(state);
    let w = usize::from(area.width);
    let digits = q.paths.len().to_string().len();
    let lines: Vec<Line> = q
        .paths
        .iter()
        .enumerate()
        .skip(scroll)
        .take(usize::from(area.height))
        .map(|(i, path)| {
            let current = q.current == Some(i);
            let title = state
                .details
                .get(path)
                .and_then(|d| d.title.clone())
                .unwrap_or_else(|| {
                    path.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                });
            // The play symbol on the current track.
            let symbol = if current {
                text::play_symbol(state.playing.state)
            } else {
                " "
            };
            // A spinner while work on it is on (an analysis, a look-up, a
            // timing), a dot while some is still to come.
            let (busy, busy_color) = match crate::core::readiness(state, path) {
                crate::core::Readiness::Working => (text::spinner(state), ACCENT),
                crate::core::Readiness::Waiting => ('·', DIM),
                crate::core::Readiness::Done => (' ', DIM),
            };
            // How far it is got ready, left of its format (where there is
            // room for its title too).
            let mut right = chip_spans(state, path);
            if w >= 56 {
                let mut icons = status_spans(state, path);
                icons.append(&mut right);
                right = icons;
            }
            let mut l = row_line(
                vec![
                    Span::styled(symbol, Style::new().fg(ACCENT)),
                    Span::styled(busy.to_string(), Style::new().fg(busy_color)),
                    Span::styled(format!("{:>digits$} ", i + 1), Style::new().fg(DIM)),
                    Span::styled(
                        title,
                        if current {
                            Style::new().fg(ACCENT)
                        } else {
                            Style::new().fg(TEXT)
                        },
                    ),
                ],
                right,
                w,
            );
            if i == cursor && state.side == Side::Queue {
                l = l.style(Style::new().bg(SELECTED).add_modifier(Modifier::BOLD));
            }
            l
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
}
