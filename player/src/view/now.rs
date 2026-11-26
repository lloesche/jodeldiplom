//! What is playing: the main screen's right column (the cover, the track's
//! info with its links and beat pulse, the summary under them), and Now
//! playing's caption ([`NowPlaying`], which the window draws over the
//! cover filling the screen: `gui::screens::now`).

use std::fmt;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::*;
use super::{keep_start, Pictures, Scene};
use crate::core::keymap::Op;
use crate::core::text::{self, Seg, Tone};
use crate::core::{Rhythm, State};
use crate::view::cells;
use crate::view::layout::MainLayout;

/// How long the pulse stays bright after a beat, in seconds.
const PULSE_SECONDS: f64 = 0.12;

/// Underlines the links in `lines` drawn in `area` that should look like
/// links now.
fn underline_links(buf: &mut Buffer, scene: &Scene, lines: &[Vec<Seg>], area: Rect) {
    for (rect, _) in crate::view::mouse::link_rects(lines, area) {
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

/// The beat pulse as it is now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pulse {
    /// Dim: between beats, the beat unclear, or not playing.
    Rest,
    /// Bright for a moment on a beat.
    Beat,
    /// Bright in the accent on the first beat of a bar.
    Bar,
}

/// The beat pulse `heard` seconds into the playing track: bright for a
/// moment on each beat (in the accent on the first beat of a bar) where
/// the beat is clear, dim otherwise.
pub fn pulse(state: &State, heard: f64) -> Pulse {
    if state.playing.state != jodel_playback::PlayState::Playing {
        return Pulse::Rest;
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
        return Pulse::Rest;
    };
    match a.beat_at(heard) {
        Some(i) if heard - a.beats[i] < PULSE_SECONDS && a.is_clear_at(heard) => {
            if a.is_downbeat(i) {
                Pulse::Bar
            } else {
                Pulse::Beat
            }
        }
        _ => Pulse::Rest,
    }
}

/// The beat pulse's look now ([`pulse`]).
fn pulse_style(scene: &Scene) -> Style {
    match pulse(scene.state, scene.heard) {
        Pulse::Rest => tone_style(Tone::Pulse),
        Pulse::Beat => Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
        Pulse::Bar => Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
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
        let segs = cells::info_lines(state, usize::from(info.width));
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
    let second = cells::wrap(&second, max)
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

/// Now playing's caption: what the window shows over the cover's bottom
/// (under the placeholder where there is no cover): the title, the artist
/// and the album (links to them in the browser), the file's format and
/// rate with the status light's word, and the tempo with its beat pulse.
#[derive(Clone, Debug, PartialEq)]
pub struct NowPlaying {
    /// The title, a link to the track ([`Tone::TitleLink`]); "Nothing
    /// playing" when nothing is.
    pub title: Seg,
    /// The artist and the album (links), a dot between; none without
    /// tags.
    pub by: Vec<Seg>,
    /// The file's format and rate (`FLAC 48`, in [`Tone::Source`]), and
    /// the status light's word (`PCM`, or what a lossy file is).
    pub chips: Vec<Seg>,
    /// The tempo and the key, the beat pulse's `♩` first
    /// ([`text::bpm_line`]).
    pub tempo: Option<Vec<Seg>>,
    /// The beat pulse now.
    pub pulse: Pulse,
}

impl NowPlaying {
    /// The caption for `state`, `heard` seconds into the playing track.
    pub fn of(state: &State, heard: f64) -> NowPlaying {
        let pulse = pulse(state, heard);
        let Some(track) = &state.playing.track else {
            return NowPlaying {
                title: text::seg("Nothing playing", Tone::Title),
                by: Vec::new(),
                chips: Vec::new(),
                tempo: None,
                pulse,
            };
        };
        let info = &track.info;
        let mut by = Vec::new();
        if let Some(artist) = &info.tags.artist {
            by.push(text::seg(artist.clone(), Tone::ArtistLink));
        }
        if let Some(album) = &info.tags.album {
            if !by.is_empty() {
                by.push(text::seg(" · ", Tone::Dim));
            }
            by.push(text::seg(album.clone(), Tone::AlbumLink));
        }
        let (_, status, tone) = text::status_light(state);
        let format = format!("{} {}", info.format_name(), text::khz_short(info.rate));
        NowPlaying {
            title: text::seg(text::track_title(info), Tone::TitleLink),
            by,
            chips: vec![text::seg(format, Tone::Source), text::seg(status, tone)],
            tempo: text::bpm_line(state),
            pulse,
        }
    }

    /// Where a piece in `tone` goes when clicked: the title, the artist
    /// and the album are links.
    pub fn link(tone: Tone) -> Option<Op> {
        text::LINKS
            .iter()
            .find(|(t, _)| *t == tone)
            .map(|&(_, op)| op)
    }
}

/// An outline of the caption, one line a piece with its kind in
/// brackets: for the snapshots.
impl fmt::Display for NowPlaying {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let joined = |segs: &[Seg]| segs.iter().map(|s| s.text.as_str()).collect::<String>();
        writeln!(f, "[title] {}", self.title.text)?;
        if !self.by.is_empty() {
            writeln!(f, "[by] {}", joined(&self.by))?;
        }
        for chip in &self.chips {
            let tone = if chip.tone == Tone::Source {
                " source"
            } else {
                ""
            };
            writeln!(f, "[chip{tone}] {}", chip.text)?;
        }
        if let Some(tempo) = &self.tempo {
            writeln!(f, "[tempo {:?}] {}", self.pulse, joined(tempo))?;
        }
        let links: Vec<String> = std::iter::once(&self.title)
            .chain(&self.by)
            .filter_map(|s| NowPlaying::link(s.tone).map(|op| format!("{} → {op:?}", s.text)))
            .collect();
        if !links.is_empty() {
            writeln!(f, "[links] {}", links.join(", "))?;
        }
        Ok(())
    }
}
