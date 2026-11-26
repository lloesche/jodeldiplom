//! Now playing's caption ([`NowPlaying`]), which the window draws over the
//! cover filling the screen (`gui::screens::now`).

use std::fmt;

use crate::core::keymap::Op;
use crate::core::text::{self, Seg, Tone};
use crate::core::State;
use crate::view::info;

/// The beat pulse as the caption shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pulse {
    /// Dim: between beats, the beat unclear, or not playing.
    Rest,
    /// Bright for a moment on a beat.
    Beat,
    /// Bright in the accent on the first beat of a bar.
    Bar,
}

impl Pulse {
    /// The beat pulse `heard` seconds into the playing track
    /// ([`info::pulse`]).
    pub fn of(state: &State, heard: f64) -> Pulse {
        match info::pulse(state, heard) {
            None => Pulse::Rest,
            Some(beat) if beat.downbeat => Pulse::Bar,
            Some(_) => Pulse::Beat,
        }
    }
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
        let pulse = Pulse::of(state, heard);
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
