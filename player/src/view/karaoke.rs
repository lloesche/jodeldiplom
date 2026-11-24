//! The Karaoke tab and Karaoke mode ([`crate::core::karaoke`]).
//!
//! The tab is cells, here: the cover, the button into the mode, the
//! sliders (the mode's controls use them too) and a few notes. The mode is
//! [`mode`]'s: one picture over the whole screen ([`KaraokePicture`]: the
//! lyrics, the count-in) that the window paints in pixels over the cover's
//! ground, its controls cells over it; a song's result is [`result`]'s.

use ratatui::buffer::Buffer;
use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Widget};
use ratatui::Frame;

use super::colors::*;
use super::{keep_start, Pictures, Scene};
use crate::core::karaoke;
use crate::core::lanes::{self, WordLines};
use crate::core::layout::SWITCH;
use crate::core::layout::{KaraokeTabLayout, SliderLayout, KARAOKE_ENTER};
use crate::core::sliders::Slider;
use crate::core::text::{self, width};
use crate::core::{Gpu, State};

mod mode;
mod result;
pub use mode::{
    break_cue, draw_mode, fill_at, line_color, lines_now, picture, points_text, scrolling, sung_on,
    timed_lines, with_start_cue, Cue, KaraokePicture, LineKind, LyricLine, Score, Start, BREAK,
    SCROLL,
};
pub use result::{
    best_words, date_words, plain_lines, result_now, verdict_words, DiplomaWords, ResultShown,
    DIPLOMA,
};

/// What to do about a microphone giving only silence, for this system.
const SILENT_HINT: &str = if cfg!(target_os = "macos") {
    "Allow it in System Settings › Privacy & Security › Microphone (Jodeldiplom, or the terminal it runs in)."
} else {
    "Check the input is not muted in the system's sound settings."
};

/// Draws the tab.
pub fn draw_tab(
    frame: &mut Frame,
    scene: &Scene,
    layout: &KaraokeTabLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    if layout.cover.width > 0 && layout.cover.height > 0 {
        pictures.cover(frame, layout.cover, false);
    }
    // The song playing, or the one the mode starts with.
    let (title, mut artist) = match &state.playing.track {
        Some(t) => (
            text::track_title(&t.info),
            t.info.tags.artist.clone().unwrap_or_default(),
        ),
        None => ("Nothing playing".into(), String::new()),
    };
    // The key it is in, once found.
    let key = crate::core::lanes::rhythmed(state, crate::core::lanes::Block::Now)
        .and_then(|r| r.analysis.key);
    if let Some(key) = key {
        if !artist.is_empty() {
            artist.push_str(" · ");
        }
        artist.push_str(&text::key_text(&key));
    }
    let max = usize::from(layout.title.width);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                keep_start(&title, max),
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(keep_start(&artist, max), Style::new().fg(DIM))),
        ]),
        layout.title,
    );
    // Greyed out until the song's lyrics are ready, why beside it.
    let why = karaoke::song_index(state).and_then(|i| karaoke::not_singable(state, i));
    button(
        frame.buffer_mut(),
        layout.enter,
        KARAOKE_ENTER,
        why.is_none(),
    );
    if let Some(why) = why {
        let x = layout.enter.right() + 2;
        let room = usize::from((layout.title.right()).saturating_sub(x));
        if room > 8 && layout.enter.height > 0 {
            frame.buffer_mut().set_string(
                x,
                layout.enter.y + layout.enter.height / 2,
                keep_start(why, room),
                Style::new().fg(DIM),
            );
        }
    }
    for s in &layout.sliders {
        slider(frame.buffer_mut(), state, s, true);
    }
    game_settings(frame.buffer_mut(), state, layout);
    super::sync::row(frame.buffer_mut(), state, layout);
    let gpu = match state.gpu {
        Gpu::Present(_) => "Here and in the mode the voice plays at its level, the rest as it is.",
        Gpu::Looking => "Still looking for a GPU for the stems.",
        Gpu::Absent => "No GPU: the voice is turned down only where its stems are kept.",
    };
    // What is known of the lyrics of the song and the next two.
    let mut lines: Vec<Line> = Vec::new();
    if let Some(first) = karaoke::song_index(state) {
        for (k, path) in state.queue.paths.iter().enumerate().skip(first).take(3) {
            let what = match karaoke::lyrics_of(state, k) {
                Some(karaoke::SongLyrics::Pending) => format!("{} looking…", text::spinner(state)),
                Some(l) => l.describe(),
                None => String::new(),
            };
            let heard = karaoke::lyrics_of(state, k).is_some_and(karaoke::SongLyrics::heard);
            let what_color = if heard { WARN } else { DIM };
            let who = if k == first { "Lyrics" } else { "  next" };
            let name = crate::core::shown_title(state, path);
            let max = usize::from(layout.notes.width).saturating_sub(width(who) + width(&what) + 5);
            // The song stands out (the one playing most); what was found of
            // its lyrics, less.
            let name_style = if k == first {
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(TEXT)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{who}  "), Style::new().fg(TEXT)),
                Span::styled(keep_start(&name, max.max(8)), name_style),
                Span::styled(format!(": {what}"), Style::new().fg(what_color)),
            ]));
            // The song's own lyrics heard by machine learning: said loudly.
            if heard && k == first {
                lines.push(Line::from(Span::styled(
                    format!("        ⚠ {}", karaoke::HEARD_WARNING),
                    Style::new().fg(WARN).add_modifier(Modifier::BOLD),
                )));
            }
        }
        lines.push(Line::default());
    }
    // The microphone: which, and how loud the singer is now (the music's
    // echo taken out).
    let mic = &state.karaoke.mic;
    let mic_line = match (&mic.name, &mic.error) {
        (Some(name), _) if mic.silent => Line::from(vec![
            Span::styled("Mic     ", Style::new().fg(TEXT)),
            Span::styled(
                format!("{name}: only silence comes in"),
                Style::new().fg(FAIL),
            ),
        ]),
        (Some(name), _) => {
            // −60 dB to 0 dB in twelve steps.
            let peak = state.mic.map_or(0.0, |m| m.singer());
            let db = 20.0 * f64::from(peak.max(1e-6)).log10();
            let [lit, unlit] = super::blocks((db + 60.0) / 60.0, 12, ACCENT);
            Line::from(vec![
                Span::styled("Mic     ", Style::new().fg(TEXT)),
                lit,
                unlit,
                Span::styled(format!("  {name}  (i: the next one)"), Style::new().fg(DIM)),
            ])
        }
        (None, Some(why)) => Line::from(vec![
            Span::styled("Mic     ", Style::new().fg(TEXT)),
            Span::styled(
                format!("not open: {why}  (i: another one)"),
                Style::new().fg(FAIL),
            ),
        ]),
        (None, None) => Line::from(Span::styled(
            format!("Mic     {} opening…", text::spinner(state)),
            Style::new().fg(DIM),
        )),
    };
    lines.push(mic_line);
    if mic.name.is_some() && mic.silent {
        lines.push(Line::from(Span::styled(
            format!("        {SILENT_HINT}"),
            Style::new().fg(DIM),
        )));
    }
    // The echo canceller: how much of the music's echo it takes out of
    // what the microphone hears.
    if mic.name.is_some() && !mic.silent {
        let echo = echo_text(state);
        lines.push(Line::from(vec![
            Span::styled("Echo    ", Style::new().fg(TEXT)),
            Span::styled(echo, Style::new().fg(DIM)),
        ]));
    }
    let words = match &state.karaoke.model {
        karaoke::ModelState::Missing => {
            "not timed: the speech model is not in this build (make puts it there)".to_owned()
        }
        karaoke::ModelState::Coming => {
            "timed once the speech model is fetched (378 MB, when first needed)".to_owned()
        }
        karaoke::ModelState::Ready => "timed against the voice by the speech model".to_owned(),
    };
    lines.push(Line::from(vec![
        Span::styled("Words   ", Style::new().fg(TEXT)),
        Span::styled(words, Style::new().fg(DIM)),
    ]));
    lines.push(Line::default());
    let notes = [
        gpu,
        "Each song waits for START, then counts in 3, 2, 1, a second each; again after a break of 8 s.",
        "The microphone only listens: it is never played; the music stays as it is.",
        "The music's echo is taken out of what it hears; your voice never is.",
        "esc in the mode comes back here.",
    ];
    lines.extend(
        notes
            .iter()
            .map(|n| Line::from(Span::styled(*n, Style::new().fg(DIM)))),
    );
    frame.render_widget(Paragraph::new(lines), layout.notes);
    if let Some(voice) = layout.voice {
        draw_voice(frame, scene, voice, pictures);
    }
    if let Some(panel) = layout.panel {
        super::sync::panel(frame.buffer_mut(), state, panel, &layout.panel_buttons);
    }
}

/// Cells a second the voice's lane shows without lyrics to fit.
const VOICE_PER: f64 = 6.0;

/// How tall the words on the tab's voice lane are, in cells: large enough
/// to read as they go by, small enough that a song sung fast does not make
/// them race.
pub const VOICE_TEXT: u16 = 2;

/// The span the voice's lane on the tab shows and where it starts: at the
/// zoom that leaves out no word of the song's lyrics, each where it is sung
/// ([`crate::core::lanes::lyrics_per_second`]), with each character `scale`
/// cells wide, the playhead a third of the way in.
pub fn voice_window(state: &State, width: u16, heard: f64, scale: u16) -> (f64, f64) {
    use crate::core::lanes::{fitted_words, lyrics_per_second, Block};
    let per = lanes::lyrics(state, Block::Now)
        .and_then(|l| lyrics_per_second(&fitted_words(&l), WordLines::KARAOKE.rows))
        .unwrap_or(VOICE_PER);
    let span = f64::from(width.max(1)) / (per * f64::from(scale.max(1)));
    let start = (heard - VOICE_LEAD * span).max(0.0);
    (start, start + span)
}
/// The share of the voice lane before the playhead.
pub const VOICE_LEAD: f64 = 1.0 / 3.0;

/// The voice's lane of the song playing, its words on it as they are sung.
fn draw_voice(frame: &mut Frame, scene: &Scene, area: Rect, pictures: &mut dyn Pictures) {
    use crate::core::lanes::{rhythmed, Block};
    use crate::core::layout::Lane;
    use crate::view::lanes::{lane_color, WavePlot};
    let state = scene.state;
    let r = rhythmed(state, Block::Now);
    let waveform = r
        .and_then(|r| r.waveform.clone())
        .filter(|w| w.stems.is_some());
    let (Some(r), Some(waveform)) = (r, waveform) else {
        let note = match (&state.playing.track, &state.gpu) {
            (None, _) => "The voice and its words show here while a song plays.",
            (_, Gpu::Absent) => "No GPU for the stems: the voice cannot be shown.",
            _ => "The voice and its words show here once the song's stems are in.",
        };
        let y = area.y + area.height / 2;
        frame.render_widget(
            Paragraph::new(Span::styled(note, Style::new().fg(DIM)))
                .alignment(HorizontalAlignment::Center),
            Rect::new(area.x, y, area.width, 1),
        );
        return;
    };
    let heard = scene.heard;
    // Larger words where the lane is tall enough for both their lines.
    let lines = WordLines::KARAOKE;
    let scale = VOICE_TEXT.min(area.height / lines.rows as u16).max(1);
    let (start, end) = voice_window(state, area.width, heard, scale);
    // The whole track's bars: the lane may be scrolled between frames.
    let a = &r.analysis;
    let downbeats = a
        .downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect();
    pictures.wave(
        frame,
        &WavePlot {
            area,
            waveform,
            lane: Lane::Vocals,
            start,
            end,
            color: lane_color(Lane::Vocals),
            downbeats,
            unclear: Vec::new(),
            selection: None,
            shown: None,
            playhead: state.playing.track.as_ref().map(|_| heard),
            cursor: None,
            cue: None,
            transition: None,
            gain: Vec::new(),
            edit: None,
            rows: (0, area.height),
            lyrics: lanes::lyrics(state, Block::Now),
            text_scale: scale,
            lines,
            follow: Some(VOICE_LEAD),
            deep: true,
            key: a.key,
        },
    );
}

/// A setting's row on the tab: its name, then its choices side by side,
/// the one chosen lit (a click on one picks it), then what it means.
#[allow(clippy::too_many_arguments)]
fn choices(
    buf: &mut Buffer,
    x: u16,
    y: u16,
    name: &str,
    items: &[(Rect, &str, bool)],
    lit: bool,
    what: &str,
    right: u16,
) {
    buf.set_string(x, y, name, Style::new().fg(TEXT));
    let mut end = x;
    for &(rect, label, chosen) in items {
        let style = if chosen && lit {
            Style::new().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
        } else if chosen {
            Style::new().fg(BG).bg(DIM)
        } else {
            Style::new().fg(DIM).bg(TRACK)
        };
        let text = format!(
            " {label:<w$}",
            w = usize::from(rect.width.saturating_sub(1))
        );
        buf.set_string(
            rect.x,
            rect.y,
            keep_start(&text, usize::from(rect.width)),
            style,
        );
        end = end.max(rect.right());
    }
    let room = usize::from(right.saturating_sub(end + 2));
    if room > 4 {
        buf.set_string(end + 2, y, keep_start(what, room), Style::new().fg(DIM));
    }
}

/// The game's settings on the tab: the game on or off, the notes shown or
/// hidden, the singing kept; each a click (or its key: g, p).
fn game_settings(buf: &mut Buffer, state: &State, layout: &KaraokeTabLayout) {
    let config = &state.config;
    let right = layout.title.right();
    let switch = |rect: Rect, on: bool| -> Vec<(Rect, &'static str, bool)> {
        let at = rect.x + 12;
        vec![(
            Rect::new(at, rect.y, SWITCH, 1),
            if on { "on" } else { "off" },
            true,
        )]
    };
    let game = config.karaoke_game;
    choices(
        buf,
        layout.game.x,
        layout.game.y,
        "Game  (g)",
        &switch(layout.game, game),
        game,
        if game {
            "your singing scored line by line against the original"
        } else {
            "just singing, nothing scored"
        },
        right,
    );
    let notes = config.karaoke_notes;
    choices(
        buf,
        layout.show_notes.x,
        layout.show_notes.y,
        "Notes (p)",
        &switch(layout.show_notes, notes),
        notes,
        if notes {
            "the song's notes run by in the mode, for you to sing on"
        } else {
            "only the words"
        },
        right,
    );
    let keep = config.karaoke_keep;
    choices(
        buf,
        layout.keep.x,
        layout.keep.y,
        "Keep",
        &switch(layout.keep, keep),
        keep,
        if keep {
            "your singing of each song saved (recordings, beside the settings), to tune the game"
        } else {
            "your singing is not saved"
        },
        right,
    );
}

/// A button: a rounded border around its label, in the accent colour.
fn button(buf: &mut Buffer, area: Rect, label: &str, lit: bool) {
    if area.width < 3 || area.height == 0 {
        return;
    }
    let color = if lit { ACCENT } else { DIM };
    if area.height >= 3 {
        Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(color))
            .render(area, buf);
    }
    let y = area.y + area.height / 2;
    let w = width(label) as u16;
    let x = area.x + area.width.saturating_sub(w) / 2;
    buf.set_string(
        x,
        y,
        label,
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    );
}

/// A slider: its label (in the accent colour when the arrow keys move
/// it), the track with its knob, and the value.
fn slider(buf: &mut Buffer, state: &State, s: &SliderLayout, show_focus: bool) {
    let focused = show_focus && state.karaoke.focus == s.slider;
    let label_style = if focused {
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(TEXT)
    };
    buf.set_string(s.row.x, s.row.y, s.slider.label(), label_style);
    let n = usize::from(s.track.width);
    if n >= 2 {
        let knob = usize::from(
            crate::core::layout::Track::columns(s.track).cell(s.slider.share(state)) - s.track.x,
        );
        let on = Style::new().fg(ACCENT);
        let off = Style::new().fg(TRACK);
        for i in 0..n {
            let (symbol, style) = match i.cmp(&knob) {
                std::cmp::Ordering::Less => ("━", on),
                std::cmp::Ordering::Equal => ("●", on.add_modifier(Modifier::BOLD)),
                std::cmp::Ordering::Greater => ("─", off),
            };
            buf.set_string(s.track.x + i as u16, s.track.y, symbol, style);
        }
    }
    buf.set_string(
        s.value.x,
        s.value.y,
        s.slider.text(state),
        Style::new().fg(TEXT),
    );
    // The voice's level means nothing without stems.
    if s.slider == Slider::Voice && !matches!(state.gpu, Gpu::Present(_)) {
        buf.set_style(s.row, Style::new().fg(DIM));
    }
}

/// The notes' colour: those sung ([`NOTE`]) and those to come.
///
/// [`NOTE`]: crate::view::lanes::NOTE
pub const NOTE_AHEAD: Color = Color::Rgb(78, 120, 146);
/// A note where the singer hit it: gold.
pub const HIT: Color = Color::Rgb(255, 190, 60);
/// The singer's own pitch: coral.
pub const SINGER: Color = Color::Rgb(240, 110, 80);

/// What the tab says of the echo canceller.
fn echo_text(state: &State) -> String {
    match (state.config.karaoke_echo, state.mic.and_then(|m| m.echo())) {
        (false, _) => "off (karaoke_echo in the settings file)".to_owned(),
        (true, None) => "on, once music plays".to_owned(),
        (true, Some((_, false, _))) => format!("{} learning the room…", text::spinner(state)),
        (true, Some((db, true, delay))) => match delay {
            Some(d) => format!(
                "the music's echo taken out, {db:.0} dB (it comes {:.0} ms late)",
                d * 1000.0
            ),
            None => format!("the music's echo taken out, {db:.0} dB"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tab says what the canceller does: off, waiting for music,
    /// learning, or how much of the echo it takes out.
    #[test]
    fn the_tab_says_what_the_canceller_does() {
        let mut state = State::new("/m".into(), crate::config::Config::default(), (120, 40));
        state.config.karaoke_echo = true;
        assert_eq!(echo_text(&state), "on, once music plays");
        let mut mic = crate::core::mic::MicReading {
            echo_on: true,
            voice: Some(0.1),
            ..Default::default()
        };
        state.mic = Some(mic);
        assert!(echo_text(&state).ends_with("learning the room…"));
        mic.removed_db = 42.0;
        mic.learnt = true;
        mic.delay = Some(0.04);
        state.mic = Some(mic);
        assert_eq!(
            echo_text(&state),
            "the music's echo taken out, 42 dB (it comes 40 ms late)"
        );
        state.config.karaoke_echo = false;
        assert!(echo_text(&state).starts_with("off"));
    }
}
