//! The Karaoke tab and Karaoke mode ([`crate::core::karaoke`]).
//!
//! The tab is [`tab`]'s model, which the window draws in its own widgets;
//! here are only its pictures: the cover, and the voice's lane under the
//! pane ([`tab_pictures`]). The mode is [`mode`]'s: one picture over the
//! whole screen ([`KaraokePicture`]: the lyrics, the count-in) that the
//! window paints in pixels over the cover's ground, its controls cells
//! over it ([`slider`]); a song's result is [`result`]'s.

use std::sync::Arc;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::Frame;

use super::colors::*;
use super::{Pictures, Scene};
use crate::core::lanes::{self, Block, WordLines};
use crate::core::sliders::Slider;
use crate::core::{Gpu, Rhythmed, State};
use crate::view::layout::{KaraokeTabLayout, SliderLayout};

mod mode;
mod result;
mod tab;
pub use mode::{
    break_cue, draw_mode, fill_at, line_color, lines_now, picture, points_text, scrolling, sung_on,
    timed_lines, with_start_cue, Cue, KaraokePicture, LineKind, LyricLine, Score, Start, BREAK,
    SCROLL,
};
pub use result::{
    best_words, date_words, plain_lines, result_now, DiplomaWords, ResultShown, DIPLOMA,
};
pub use tab::{
    keys_line, track_id, Control, Enter, Hint, LyricsRow, Mic, SliderRow, Switch, Tab, ENTER,
    MEASURE,
};

/// The tab's pictures: the playing track's cover, and the voice's lane
/// with its words, under the pane (where the other tabs have the
/// spectrum), while it has stems to show.
pub fn tab_pictures(
    frame: &mut Frame,
    scene: &Scene,
    layout: &KaraokeTabLayout,
    pictures: &mut dyn Pictures,
) {
    if layout.cover.width > 0 && layout.cover.height > 0 {
        // At its top left, half a cell down: level with the text beside it.
        pictures.cover(frame, layout.cover, true);
    }
    if let (Some(area), Some((r, waveform))) = (layout.voice, voice_shown(scene.state)) {
        voice_lane(frame, scene, area, r, waveform, pictures);
    }
}

/// The playing song's analysis and its waveform with stems, where the
/// voice's lane has them to show.
fn voice_shown(state: &State) -> Option<(&Rhythmed, Arc<jodel_rhythm::waveform::Waveform>)> {
    let r = lanes::rhythmed(state, Block::Now)?;
    let waveform = r.waveform.clone().filter(|w| w.stems.is_some())?;
    Some((r, waveform))
}

/// What the voice's lane says while it has nothing to show (none while it
/// shows the voice).
pub fn voice_note(state: &State) -> Option<&'static str> {
    if voice_shown(state).is_some() {
        return None;
    }
    Some(match (&state.playing.track, &state.gpu) {
        (None, _) => "The voice and its words show here while a song plays.",
        (_, Gpu::Absent) => "No GPU for the stems: the voice cannot be shown.",
        _ => "The voice and its words show here once the song's stems are in.",
    })
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

/// The voice's lane of the song playing in `area`, its words on it as
/// they are sung.
fn voice_lane(
    frame: &mut Frame,
    scene: &Scene,
    area: Rect,
    r: &Rhythmed,
    waveform: Arc<jodel_rhythm::waveform::Waveform>,
    pictures: &mut dyn Pictures,
) {
    use crate::core::lanes::Lane;
    use crate::view::lanes::{lane_color, WavePlot};
    use jodel_playback::Stem;
    let state = scene.state;
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
            lane: Lane::Stem(Stem::Vocals),
            start,
            end,
            color: lane_color(Lane::Stem(Stem::Vocals)),
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

/// A slider of the mode's controls: its label (in the accent colour when
/// the arrow keys move it), the track with its knob, and the value.
fn slider(buf: &mut Buffer, state: &State, s: &SliderLayout) {
    let focused = state.karaoke.focus == s.slider;
    let label_style = if focused {
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(TEXT)
    };
    buf.set_string(s.row.x, s.row.y, s.slider.label(), label_style);
    let n = usize::from(s.track.width);
    if n >= 2 {
        let knob = usize::from(
            crate::view::layout::Track::columns(s.track).cell(s.slider.share(state)) - s.track.x,
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
pub const NOTE_AHEAD: [u8; 3] = [78, 120, 146];
/// A note where the singer hit it: gold.
pub const HIT: [u8; 3] = [255, 190, 60];
/// The singer's own pitch: coral.
pub const SINGER: [u8; 3] = [240, 110, 80];
