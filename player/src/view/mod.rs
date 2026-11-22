//! The ratatui widgets that draw the [`State`]: the same code for the
//! terminal and the window.
//!
//! Pictures (the cover and the spectrum) go through [`Pictures`]: the
//! terminal draws them into cells (an image protocol or half blocks for
//! the cover, braille dots for the spectrum, [`spectrum_cells`]); the
//! window leaves their cells empty and paints pixels there after the frame
//! is drawn. The spectrum's axis labels are cells in both ([`plot`]).

pub mod desk;
pub mod karaoke;
pub mod lanes;
pub mod plot;
pub mod setup;
pub mod sync;
pub mod transition;

pub use plot::{spectrum_cells, Plot};

use ratatui::buffer::Buffer;
use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Widget};
use ratatui::Frame;

use jodel_playback::Headroom;

use crate::browser::EntryKind;
use crate::core::layout::{
    self, Bottom, HelpLayout, Layout, MainLayout, NowLayout, SettingsLayout, SETTINGS_LABEL,
};
use crate::core::text::{self, clock, khz_short, width, Seg, Tone};
use crate::core::{
    browser_rows, device_window, help_shown, list_cursor, palette_window, row_entry, Gpu, Level,
    Rhythm, Row, Screen, SettingsRow, Side, State,
};
use crate::spectrum::SpectrumFrame;

/// Colours.
pub mod colors {
    use ratatui::style::Color;

    /// The window background.
    pub const BG: Color = Color::Rgb(17, 17, 21);
    /// Highlights and the play head: the Okabe–Ito yellow, apart from the
    /// drums' orange.
    pub const ACCENT: Color = Color::Rgb(240, 228, 66);
    /// Secondary text and borders.
    pub const DIM: Color = Color::Rgb(120, 120, 132);
    /// Text.
    pub const TEXT: Color = Color::Rgb(222, 222, 228);
    /// Borders.
    pub const BORDER: Color = Color::Rgb(64, 64, 74);
    /// The selected row.
    pub const SELECTED: Color = Color::Rgb(46, 46, 58);
    /// Buttons.
    pub const BUTTON: Color = Color::Rgb(38, 38, 47);
    /// Good news (bit-perfect, done).
    pub const GREEN: Color = Color::Rgb(46, 204, 113);
    /// Warnings.
    pub const WARN: Color = Color::Rgb(240, 160, 60);
    /// Errors.
    pub const FAIL: Color = Color::Rgb(235, 80, 80);
    /// The file's rate, and the spectrum's curve.
    pub const SOURCE: Color = Color::Rgb(150, 160, 200);
    /// Folders.
    pub const FOLDER: Color = Color::Rgb(130, 180, 255);
    /// The unplayed part of the progress bar.
    pub const TRACK: Color = Color::Rgb(60, 60, 70);
    /// The spectrum's grid.
    pub const GRID: Color = Color::Rgb(78, 78, 92);
}

use colors::*;

/// A spectrum zone: frequencies up to `up_to_hz` belong to a stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    /// The zone's top frequency.
    pub up_to_hz: f64,
    /// Its colour.
    pub color: Color,
}

/// Draws the pictures: the cover and the spectrum.
pub trait Pictures {
    /// The cover (or a placeholder) into `area`. With `below_line` the
    /// area's first row has a border line through its middle (the list
    /// pane's top): the cover starts under that line, half a cell down,
    /// at the left; otherwise it is centred.
    fn cover(&mut self, frame: &mut Frame, area: Rect, background: Color, below_line: bool);
    /// The spectrum's curve and grid into `plot.area` (its labels are
    /// drawn as cells by the caller).
    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot);
    /// Whether text can go over the cover just drawn: the window can; a
    /// terminal's graphics would be cut by the cells drawn over them.
    fn text_over_cover(&self) -> bool {
        false
    }
    /// The cells of `caption` are text over the cover in `cover`: the
    /// window darkens the cover behind them so they read on any cover.
    fn caption(&mut self, _cover: Rect, _caption: Rect) {}
    /// How many cells tall (and each character as many wide) the words on
    /// a lane may be drawn ([`lanes::WavePlot::text_scale`]): one in cells,
    /// more where the frontend draws text at any size.
    fn lane_text_scale(&self) -> u16 {
        1
    }
    /// A lane's waveform ([`lanes::WavePlot`]), on the Transition tab or the
    /// Karaoke tab's voice lane: in braille cells unless the frontend draws
    /// it in pixels.
    fn wave(&mut self, frame: &mut Frame, plot: &lanes::WavePlot) {
        lanes::wave_cells(frame.buffer_mut(), plot);
    }
    /// A Transition block's bar numbers ([`transition::RulerPlot`]): in cells
    /// unless the frontend draws them where the bars are, to the pixel.
    fn ruler(&mut self, frame: &mut Frame, plot: &transition::RulerPlot) {
        transition::ruler_cells(frame.buffer_mut(), plot);
    }
    /// Karaoke mode's picture ([`karaoke::KaraokePicture`]): in cells
    /// unless the frontend paints it in pixels.
    fn karaoke(&mut self, frame: &mut Frame, picture: &karaoke::KaraokePicture) {
        karaoke::karaoke_cells(frame.buffer_mut(), picture);
    }
}

/// How the title, artist and album links look.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Links {
    /// Underlined, as the device (the terminal).
    #[default]
    Underlined,
    /// Underlined while the pointer is over them (the window); the
    /// pointer's cell, if it is in the window.
    OnHover(Option<(u16, u16)>),
}

/// Everything a frame shows.
pub struct Scene<'a> {
    /// The UI state.
    pub state: &'a State,
    /// The spectrum heard now.
    pub spectrum: &'a SpectrumFrame,
    /// The background: [`colors::BG`] in the window, the terminal's own
    /// (`Color::Reset`) in a terminal.
    pub background: Color,
    /// How the links look.
    pub links: Links,
    /// Seconds into the playing track as heard now ([`crate::app::App::heard`]),
    /// for the beat pulse.
    pub heard: f64,
}

/// Underlines the links in `lines` drawn in `area` that should look like
/// links now.
fn underline_links(buf: &mut Buffer, scene: &Scene, lines: &[Vec<Seg>], area: Rect) {
    for (rect, _) in crate::core::mouse::link_rects(lines, area) {
        let shown = match scene.links {
            Links::Underlined => true,
            Links::OnHover(pointer) => {
                pointer.is_some_and(|(x, y)| rect.contains(ratatui::layout::Position::new(x, y)))
            }
        };
        if shown {
            buf.set_style(rect, Style::new().add_modifier(Modifier::UNDERLINED));
        }
    }
}

fn tone_style(tone: Tone) -> Style {
    let s = Style::new();
    match tone {
        Tone::Plain => s.fg(TEXT),
        Tone::Title | Tone::TitleLink => s.fg(TEXT).add_modifier(Modifier::BOLD),
        Tone::Accent | Tone::ArtistLink => s.fg(ACCENT),
        Tone::Dim | Tone::AlbumLink => s.fg(DIM),
        Tone::Source => s.fg(SOURCE).add_modifier(Modifier::BOLD),
        Tone::Device => s.fg(TEXT).add_modifier(Modifier::UNDERLINED),
        Tone::Good => s.fg(GREEN),
        Tone::Warn => s.fg(WARN),
        Tone::Pulse => s.fg(DIM),
    }
}

fn line(segs: &[Seg]) -> Line<'static> {
    Line::from(
        segs.iter()
            .map(|s| Span::styled(s.text.clone(), tone_style(s.tone)))
            .collect::<Vec<_>>(),
    )
}

/// How long the pulse stays bright after a beat, in seconds.
const PULSE_SECONDS: f64 = 0.12;

/// The beat pulse's look now: bright for a moment on each beat of the
/// playing track (in the accent colour on the first beat of a bar) where
/// the beat is clear, dim otherwise.
fn pulse_style(scene: &Scene) -> Style {
    let state = scene.state;
    let dim = tone_style(Tone::Pulse);
    if state.playing.state != jodel_playback::PlayState::Playing {
        return dim;
    }
    let analysis = state
        .playing
        .track
        .as_ref()
        .and_then(|t| state.queue.paths.get(t.index))
        .and_then(|p| match state.rhythm.get(p) {
            Some(Rhythm::Ready(r)) => Some(&r.analysis),
            _ => None,
        });
    let Some(a) = analysis else {
        return dim;
    };
    let t = scene.heard;
    match a.beat_at(t) {
        Some(i) if t - a.beats[i] < PULSE_SECONDS && a.is_clear_at(t) => {
            let color = if a.is_downbeat(i) { ACCENT } else { TEXT };
            Style::new().fg(color).add_modifier(Modifier::BOLD)
        }
        _ => dim,
    }
}

/// [`line`] with the beat pulse drawn as it looks now.
fn line_at(segs: &[Seg], scene: &Scene) -> Line<'static> {
    Line::from(
        segs.iter()
            .map(|s| {
                let style = if s.tone == Tone::Pulse {
                    pulse_style(scene)
                } else {
                    tone_style(s.tone)
                };
                Span::styled(s.text.clone(), style)
            })
            .collect::<Vec<_>>(),
    )
}

/// Draws the whole screen.
pub fn draw(frame: &mut Frame, scene: &Scene, pictures: &mut dyn Pictures) {
    let area = frame.area();
    let bg = Style::new().bg(scene.background).fg(TEXT);
    Block::new().style(bg).render(area, frame.buffer_mut());
    // The layout is in the state's size; a frame of another size (the
    // terminal resized since the state last heard of it) stays blank until
    // the state catches up, as nothing laid out for it would fit.
    if (area.width, area.height) != scene.state.size {
        return;
    }
    match layout::compute(scene.state) {
        Layout::TooSmall(area) => {
            let text = format!(
                "Make the window larger ({} × {} at least)",
                layout::MIN_SIZE.0,
                layout::MIN_SIZE.1
            );
            let y = area.y + area.height / 2;
            frame.render_widget(
                Paragraph::new(text)
                    .style(Style::new().fg(DIM))
                    .alignment(HorizontalAlignment::Center),
                Rect::new(area.x, y, area.width, 1),
            );
        }
        Layout::Main(m) => draw_main(frame, scene, &m, pictures),
        Layout::NowPlaying(n) => draw_now(frame, scene, &n, pictures),
        Layout::Settings(s) => draw_settings(frame, scene, &s),
        Layout::Help(h) => draw_help(frame, scene, &h),
        Layout::Karaoke(k) => karaoke::draw_mode(frame, scene, &k, pictures),
    }
    if let Some(bottom) = layout::compute(scene.state).bottom() {
        draw_bottom(frame, scene, bottom, pictures);
    }
}

fn bordered(title: Line<'static>) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .title(title)
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

/// Shortens `text` to `max` cells, keeping its start.
fn keep_start(text: &str, max: usize) -> String {
    if width(text) <= max {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 1;
    for c in text.chars() {
        let w = width(c.encode_utf8(&mut [0; 4]));
        if used + w > max {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push('…');
    out
}

fn draw_main(frame: &mut Frame, scene: &Scene, m: &MainLayout, pictures: &mut dyn Pictures) {
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
        Side::Mixer => desk::FOOTER.into(),
    };
    let block = bordered(Line::from(title)).title_bottom(Line::from(Span::styled(
        format!(" {footer} "),
        Style::new().fg(DIM),
    )));
    frame.render_widget(block, m.list);
    if state.palette.is_some() {
        draw_palette(frame, state, m.rows);
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
            Side::Mixer => {
                if let Some(layout) = &m.desk {
                    desk::draw(frame.buffer_mut(), state, layout);
                }
            }
        }
    }
    if let (Some(rect), Some(filter)) = (m.filter, &state.browser.filter) {
        let shown = browser_rows(state)
            .iter()
            .filter(|r| !matches!(r, Row::SearchBelow))
            .count();
        let mut spans = vec![
            Span::styled("/ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(filter.text.clone(), Style::new().fg(TEXT)),
        ];
        if filter.editing {
            spans.push(Span::styled("▏", Style::new().fg(ACCENT)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), rect);
        let count = format!("{shown} shown");
        let w = width(&count) as u16;
        if rect.width > w + 10 {
            frame.render_widget(
                Paragraph::new(Span::styled(count, Style::new().fg(DIM))),
                Rect::new(rect.right() - w, rect.y, w, 1),
            );
        }
    }
    if let Some(cover) = m.cover {
        pictures.cover(frame, cover, scene.background, true);
    }
    if let Some(info) = m.info {
        let segs = text::info_lines(state, usize::from(info.width));
        let lines: Vec<Line> = segs.iter().map(|l| line_at(l, scene)).collect();
        frame.render_widget(Paragraph::new(lines), info);
        underline_links(frame.buffer_mut(), scene, &segs, info);
    }
    if let Some(mini) = m.mini {
        draw_mini(frame, state, mini);
    }
}

fn draw_mini(frame: &mut Frame, state: &State, area: Rect) {
    let max = usize::from(area.width);
    let Some(track) = &state.playing.track else {
        frame.render_widget(
            Paragraph::new(Span::styled("Nothing playing", Style::new().fg(DIM))),
            area,
        );
        return;
    };
    let mut first = format!(
        "{} {}",
        text::play_symbol(state.playing.state),
        text::track_title(&track.info)
    );
    if let Some(artist) = &track.info.tags.artist {
        first.push_str(" — ");
        first.push_str(artist);
    }
    let (light, label, tone) = text::status_light(state);
    let mut second = vec![Seg {
        text: format!("{light} {label}  "),
        tone,
    }];
    second.extend(text::signal_path(state).unwrap_or_default());
    let second = text::wrap(&second, max)
        .into_iter()
        .next()
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                keep_start(&first, max),
                tone_style(Tone::Title),
            )),
            line(&second),
        ]),
        area,
    );
}

fn row_line(left: Vec<Span<'static>>, right: Vec<Span<'static>>, w: usize) -> Line<'static> {
    let right_w: usize = right.iter().map(|s| width(&s.content)).sum();
    let room = w.saturating_sub(right_w + 1);
    let mut used = 0;
    let mut spans = Vec::new();
    for span in left {
        let sw = width(&span.content);
        if used + sw > room {
            let cut = keep_start(&span.content, room.saturating_sub(used));
            used += width(&cut);
            spans.push(Span::styled(cut, span.style));
            break;
        }
        used += sw;
        spans.push(span);
    }
    if right_w > 0 && w > right_w {
        spans.push(Span::raw(" ".repeat(w - right_w - used)));
        spans.extend(right);
    }
    Line::from(spans)
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
    use crate::core::{LyricsFacts, QueueFacts, Rhythm, RhythmFacts};
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
    } = facts;
    // Heard by machine learning (no trusted source): lit in orange.
    let known = state.karaoke.lyrics.get(path);
    let heard = lyrics == Some(LyricsFacts::Heard) || known.is_some_and(SongLyrics::heard);
    let lyrics = heard
        || matches!(lyrics, Some(LyricsFacts::Timed | LyricsFacts::Untimed))
        || matches!(known, Some(SongLyrics::Found { .. }));
    let has = [
        key == Some(true),
        stems == Some(true),
        tempo == Some(true),
        lyrics,
    ];
    let mut out = Vec::new();
    for (k, (icon, has)) in ["♯", "▥", "♩", "¶"].into_iter().zip(has).enumerate() {
        let color = match (has, k == 3 && heard) {
            (true, true) => WARN,
            (true, false) => ACCENT,
            (false, _) => DIM,
        };
        out.push(Span::styled(icon, Style::new().fg(color)));
        out.push(Span::raw(" "));
    }
    out.push(Span::raw(" "));
    out
}
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
            .wrap(ratatui::widgets::Wrap { trim: true }),
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

fn draw_palette(frame: &mut Frame, state: &State, area: Rect) {
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

/// Now playing: the cover filling the screen and the caption at its
/// bottom left, over the cover where the frontend can draw text there,
/// else under it.
fn draw_now(frame: &mut Frame, scene: &Scene, n: &NowLayout, pictures: &mut dyn Pictures) {
    let over = pictures.text_over_cover();
    let cover = if over { n.cover } else { n.cover_above };
    let caption = n.caption;
    if cover.width > 0 && cover.height > 0 {
        pictures.cover(frame, cover, scene.background, false);
    }
    if caption.width == 0 || caption.height == 0 {
        return;
    }
    if over {
        pictures.caption(cover, caption);
    }
    let segs = text::caption_lines(scene.state, usize::from(caption.width));
    let lines: Vec<Line> = segs.iter().map(|l| line_at(l, scene)).collect();
    frame.render_widget(Paragraph::new(lines), caption);
    underline_links(frame.buffer_mut(), scene, &segs, caption);
}

fn rate_cap_label(cap: Option<u32>) -> String {
    match cap {
        None => "none".into(),
        Some(rate) => format!("{} kHz", khz_short(rate)),
    }
}

fn draw_settings(frame: &mut Frame, scene: &Scene, s: &SettingsLayout) {
    let state = scene.state;
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
        let label = format!("{marker}{}", row.label());
        let label_rect = Rect::new(
            line_rect.rect.x,
            line_rect.rect.y,
            SETTINGS_LABEL.min(line_rect.rect.width),
            1,
        );
        frame.render_widget(Paragraph::new(Span::styled(label, label_style)), label_rect);
        let value = match row {
            SettingsRow::Device => format!("{}  ▾", state.device_label()),
            SettingsRow::RateCap => {
                let used = state.effective_max_rate();
                if used == state.config.max_rate {
                    rate_cap_label(used)
                } else {
                    format!(
                        "{} (saved: {}, below this device)",
                        rate_cap_label(used),
                        rate_cap_label(state.config.max_rate)
                    )
                }
            }
            SettingsRow::Headroom => match state.config.headroom {
                Headroom::NONE => "none (lossy files -1.5 dB, for their overs)".to_owned(),
                headroom if headroom < jodel_playback::LOSSY_HEADROOM => {
                    format!("{headroom} where not bit-perfect, lossy files -1.5 dB")
                }
                headroom => format!("{headroom} where not bit-perfect"),
            },
            SettingsRow::Resample => if state.config.resample {
                "on: when no rate carries the file bit-perfect"
            } else {
                "off: such files are skipped"
            }
            .to_owned(),
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
                let mark = match (c.beat_voting, c.beat_methods.contains(method)) {
                    (true, true) => "[x]",
                    (true, false) => "[ ]",
                    (false, _) if c.beat_method == *method => "(●)",
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
                (Some(m), _) if m.failure.is_some() => {
                    "the move stopped: enter tries again".to_owned()
                }
                (Some(m), _) if !m.finished => format!(
                    "{} {:.0} % moving to {}",
                    bar_text(m.share, 10),
                    m.share * 100.0,
                    m.to.display()
                ),
                (_, Some(dir)) => format!("{}  ▾", dir.display()),
                (_, None) => "none  ▾".to_owned(),
            },
        };
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

/// A level as `n` blocks, `share` of them lit (▮) in `on`, the rest
/// unlit (▯): the microphone's meter, a ring's fill, the device's volume.
pub(crate) fn blocks(share: f64, n: usize, on: Color) -> [Span<'static>; 2] {
    let lit = ((share.clamp(0.0, 1.0) * n as f64).round() as usize).min(n);
    [
        Span::styled("▮".repeat(lit), Style::new().fg(on)),
        Span::styled("▯".repeat(n - lit), Style::new().fg(TRACK)),
    ]
}

/// A bar of `width` cells, `share` of it filled.
fn bar_text(share: f64, width: usize) -> String {
    let filled = ((share.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
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

/// The help screen's lines, as its search leaves them (the same count as
/// [`crate::core::help_line_count`]).
pub fn help_lines(state: &State) -> Vec<Line<'static>> {
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

fn draw_help(frame: &mut Frame, scene: &Scene, h: &HelpLayout) {
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
    let state = scene.state;
    let lines: Vec<Line> = help_lines(state)
        .into_iter()
        .skip(state.help_scroll)
        .collect();
    if let (Some(rect), Some(search)) = (h.search, &state.help_search) {
        // As the browser's filter line: what is typed, and how many keys
        // it leaves.
        let shown: usize = help_shown(state).iter().map(|s| s.lines.len()).sum();
        let mut spans = vec![
            Span::styled("/ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(search.text.clone(), Style::new().fg(TEXT)),
        ];
        if search.editing {
            spans.push(Span::styled("▏", Style::new().fg(ACCENT)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), rect);
        let count = format!("{shown} shown");
        let w = width(&count) as u16;
        if rect.width > w + 10 {
            frame.render_widget(
                Paragraph::new(Span::styled(count, Style::new().fg(DIM))),
                Rect::new(rect.right() - w, rect.y, w, 1),
            );
        }
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

/// The spectrum's zones for the track heard: one, the whole range.
pub fn zones(_state: &State) -> Vec<Zone> {
    vec![Zone {
        up_to_hz: f64::MAX,
        color: SOURCE,
    }]
}

/// The colour of a frequency.
pub fn zone_color(zones: &[Zone], hz: f64) -> Color {
    zones
        .iter()
        .find(|z| hz <= z.up_to_hz)
        .or(zones.last())
        .map_or(SOURCE, |z| z.color)
}

/// The time heard (right-aligned in `elapsed`), the seek bar filling in the
/// signal path's colour with its head, and the length; a drag shows where
/// it would go.
pub(crate) fn draw_progress(
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

fn draw_bottom(frame: &mut Frame, scene: &Scene, b: &Bottom, pictures: &mut dyn Pictures) {
    let state = scene.state;
    if let Some(area) = b.spectrum {
        let plot = Plot::new(area, b.labels, scene.spectrum, zones(state));
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

/// Cover and spectrum drawn into cells, for terminals without images and
/// for tests: the cover as a dim placeholder, the spectrum with
/// [`spectrum_cells`].
pub struct CellPictures;

impl Pictures for CellPictures {
    fn cover(&mut self, frame: &mut Frame, area: Rect, _background: Color, _below_line: bool) {
        placeholder(frame, area);
    }

    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot) {
        spectrum_cells(frame.buffer_mut(), plot);
    }
}

/// The box drawn where there is no cover.
pub fn placeholder(frame: &mut Frame, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height > 0 {
        frame.render_widget(
            Paragraph::new(Span::styled("♪", Style::new().fg(BORDER)))
                .alignment(HorizontalAlignment::Center),
            Rect::new(inner.x, inner.y + inner.height / 2, inner.width, 1),
        );
    }
}

/// The text of a buffer, one line per row (for text dumps and tests).
pub fn buffer_text(buffer: &Buffer) -> String {
    let mut out = String::new();
    for y in 0..buffer.area.height {
        let mut row = String::new();
        let mut skip = 0;
        for x in 0..buffer.area.width {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            let symbol = buffer[(x, y)].symbol();
            row.push_str(symbol);
            skip = width(symbol).saturating_sub(1);
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}
