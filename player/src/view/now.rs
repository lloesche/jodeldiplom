//! What is playing: the main screen's right column (the cover, the track's
//! info with its links and beat pulse, the summary under them) and Now
//! playing (the cover filling the screen, the caption over it).

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::*;
use super::{keep_start, Pictures, Scene};
use crate::core::layout::{MainLayout, NowLayout};
use crate::core::text::{self, Seg, Tone};
use crate::core::{Rhythm, State};

/// How long the pulse stays bright after a beat, in seconds.
const PULSE_SECONDS: f64 = 0.12;

/// Underlines the links in `lines` drawn in `area` that should look like
/// links now.
fn underline_links(buf: &mut Buffer, scene: &Scene, lines: &[Vec<Seg>], area: Rect) {
    for (rect, _) in crate::core::mouse::link_rects(lines, area) {
        let shown = scene
            .pointer
            .is_some_and(|(x, y)| rect.contains(ratatui::layout::Position::new(x, y)));
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

/// The main screen's right column: the cover, the track's info, the
/// summary under them; each where it is laid out.
pub(super) fn draw_column(
    frame: &mut Frame,
    scene: &Scene,
    m: &MainLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    if let Some(cover) = m.cover {
        pictures.cover(frame, cover, true);
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

/// Now playing: the cover filling the screen and the caption at its
/// bottom left, over the cover where the frontend can draw text there,
/// else under it.
pub(super) fn draw_now(
    frame: &mut Frame,
    scene: &Scene,
    n: &NowLayout,
    pictures: &mut dyn Pictures,
) {
    let over = pictures.text_over_cover();
    let cover = if over { n.cover } else { n.cover_above };
    let caption = n.caption;
    if cover.width > 0 && cover.height > 0 {
        pictures.cover(frame, cover, false);
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
