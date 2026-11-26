//! Text the screens show, whatever lays it out: rates, times, the signal
//! path, the status light and the playing track's links.

use jodel_playback::{PlayState, Stem};

use crate::core::keymap::Op;
use crate::core::{Rhythm, State};

/// A rate in kHz without the unit: `44.1`, `48`, `352.8`.
pub fn khz_short(rate: u32) -> String {
    let khz = f64::from(rate) / 1000.0;
    if (khz - khz.round()).abs() < 0.05 {
        format!("{khz:.0}")
    } else {
        format!("{khz:.1}")
    }
}

/// Milliseconds with their sign: `+30 ms`, `−20 ms`, `0 ms`.
pub fn signed_ms(ms: i32) -> String {
    match ms.signum() {
        1 => format!("+{ms} ms"),
        -1 => format!("−{} ms", ms.unsigned_abs()),
        _ => "0 ms".to_owned(),
    }
}

/// A time as `m:ss`, or `h:mm:ss` from an hour on.
pub fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).floor() as u64;
    let (h, m, s) = (total / 3600, total / 60 % 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// How a piece of text is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// Ordinary text.
    Plain,
    /// A title (bold).
    Title,
    /// The artist.
    Accent,
    /// Secondary text.
    Dim,
    /// The file's rate.
    Source,
    /// The output device (clickable).
    Device,
    /// The playing track's title (a link to it in the browser).
    TitleLink,
    /// Its artist (a link to the artist folder).
    ArtistLink,
    /// Its album (a link to the album folder).
    AlbumLink,
    /// A warning (a resampling step, samples clipped).
    Warn,
    /// The beat pulse: drawn brighter for a moment on each beat.
    Pulse,
}

/// A piece of coloured text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    /// The text.
    pub text: String,
    /// Its colour.
    pub tone: Tone,
}

/// A piece of `text` in `tone`.
pub fn seg(text: impl Into<String>, tone: Tone) -> Seg {
    Seg {
        text: text.into(),
        tone,
    }
}

/// The signal path of the track being heard, for example `file 48 → DAC`,
/// or with a resampler `file 44.1 → resample 48 (160/147) → DAC`. A lossy
/// file's ends with what it was decoded from: `file 44.1 → DAC, decoded
/// from MP3`. No verdict on bit-perfection: the system's mixer, after the
/// device's name, may resample and sets the volume.
pub fn signal_path(state: &State) -> Option<Vec<Seg>> {
    let track = state.playing.track.as_ref()?;
    let route = &track.route;
    let arrow = || seg(" → ", Tone::Dim);
    let input = khz_short(route.input_rate);
    let mut out = vec![seg(format!("file {input}"), Tone::Source)];
    if let Some(step) = route.resample {
        out.push(arrow());
        out.push(seg(
            format!("resample {} ({})", khz_short(step.to), step.detail()),
            Tone::Warn,
        ));
    }
    out.push(arrow());
    out.push(seg(track.device.clone(), Tone::Device));
    if let Some(codec) = route.decoded_from {
        out.push(seg(format!(", decoded from {codec}"), Tone::Dim));
    }
    if state.playing.clipped > 0 {
        out.push(seg(
            format!(", {} samples clipped", state.playing.clipped),
            Tone::Warn,
        ));
    }
    Some(out)
}

/// The status light: symbol, label, tone.
pub fn status_light(state: &State) -> (&'static str, String, Tone) {
    let playing = &state.playing;
    let Some(track) = &playing.track else {
        return ("○", "nothing playing".into(), Tone::Dim);
    };
    if track.info.is_lossy() {
        return ("○", lossy_label(&track.info), Tone::Dim);
    }
    ("○", "PCM".into(), Tone::Dim)
}

/// What the status light says for a lossy file: the codec, the bit rate
/// when known, and that it is lossy (`MP3 320 kbps · lossy`).
pub fn lossy_label(info: &jodel_playback::TrackInfo) -> String {
    match info.bitrate {
        Some(bitrate) => format!(
            "{} {} kbps · lossy",
            info.format_name(),
            (bitrate + 500) / 1000
        ),
        None => format!("{} · lossy", info.format_name()),
    }
}

/// A stem button by the transport buttons: ` drums `, or ` D ` on a
/// narrow screen.
pub fn stem_button(stem: Stem, wide: bool) -> String {
    let name = stem.id();
    if wide {
        format!(" {name} ")
    } else {
        format!(" {} ", name[..1].to_uppercase())
    }
}

/// The play state symbol: `▶`, `▮▮` or `■`.
pub fn play_symbol(state: PlayState) -> &'static str {
    match state {
        PlayState::Playing => "▶",
        PlayState::Paused => "▮▮",
        PlayState::Stopped => "■",
    }
}

/// The link tones and where each goes.
pub const LINKS: [(Tone, Op); 3] = [
    (Tone::TitleLink, Op::GoTrack),
    (Tone::ArtistLink, Op::GoArtist),
    (Tone::AlbumLink, Op::GoAlbum),
];

/// A tempo for people: whole when it is within 0.05 of a whole number,
/// else with one decimal.
pub fn bpm_text(bpm: f64) -> String {
    if (bpm - bpm.round()).abs() < 0.05 {
        format!("{bpm:.0}")
    } else {
        format!("{bpm:.1}")
    }
}

/// The playing track's tempo, and its key once found (`· E major (12B)`):
/// `♩ 128 BPM` when steady; `♩ 98 BPM · 92–104`
/// when it varies (the tempo here, then the range), or `♩ 92–104 BPM`
/// where the beat is unclear; `♩ … BPM` while it is analysed; `♩ – BPM`
/// when the beat is clear nowhere or the file could not be analysed.
/// `None` with the tempo off or nothing playing. The `♩` is the beat
/// pulse ([`Tone::Pulse`]).
pub fn bpm_line(state: &State) -> Option<Vec<Seg>> {
    if !state.config.bpm {
        return None;
    }
    let track = state.playing.track.as_ref()?;
    let path = state.queue.paths.get(track.index)?;
    let text = match state.rhythm.get(path) {
        None | Some(Rhythm::Pending) => "… BPM".to_owned(),
        Some(Rhythm::Failed) => "– BPM".to_owned(),
        Some(Rhythm::Ready(r)) => match r.analysis.summary {
            None => "– BPM".to_owned(),
            Some(s) if s.steady => format!("{} BPM", bpm_text(s.bpm)),
            Some(s) => {
                let range = format!("{:.0}–{:.0}", s.low, s.high);
                let at = state.playing.position;
                match r.analysis.bpm_at(at).filter(|_| r.analysis.is_clear_at(at)) {
                    Some(now) => format!("{:.0} BPM · {range}", now),
                    None => format!("{range} BPM"),
                }
            }
        },
    };
    let key = match state.rhythm.get(path) {
        Some(Rhythm::Ready(r)) => r.analysis.key.map(|k| format!(" · {}", key_text(&k))),
        _ => None,
    };
    Some(vec![
        seg("♩", Tone::Pulse),
        seg(format!(" {text}{}", key.unwrap_or_default()), Tone::Dim),
    ])
}

/// A key as shown: `E major (12B)`, its place on the Camelot wheel after.
pub fn key_text(key: &jodel_rhythm::key::Key) -> String {
    format!("{} ({})", key.name(), key.camelot())
}

/// The name a track is shown by: its title tag, else its file name.
pub fn track_title(info: &jodel_playback::TrackInfo) -> String {
    info.tags
        .title
        .clone()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| {
            info.path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
}

/// A spinner's glyphs, in turn.
pub const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A spinner's glyph for something still loading, turning ten times a
/// second by the clock last drawn at.
pub fn spinner(state: &State) -> char {
    let ms = state.drawn_ms.unwrap_or(state.now_ms);
    SPINNER[(ms / 100 % 10) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_clocks() {
        assert_eq!(khz_short(44_100), "44.1");
        assert_eq!(khz_short(48_000), "48");
        assert_eq!(khz_short(352_800), "352.8");
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(61.9), "1:01");
        assert_eq!(clock(3_725.0), "1:02:05");
    }
}
