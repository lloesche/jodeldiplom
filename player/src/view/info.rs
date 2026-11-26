//! The main screen's right column under the cover, as a model ([`Info`]):
//! the playing track's title, artist and album (links to where it is in
//! the browser), what the file is and the way it takes to the device as
//! tags (the status light, the file's rate, a resampling step, the
//! device, whose setting a click opens, what it was decoded from, samples
//! clipped), and its tempo and key with the beat pulse ([`pulse`]). Where
//! the window is too narrow for the column, the same model makes the
//! two-line summary under the list: the play state, the title and the
//! artist, then the tags. The window draws both
//! ([`crate::gui::screens::right`]).

use std::fmt;

use jodel_playback::PlayState;

use crate::core::keymap::{self, Context, Op};
use crate::core::text::{self, Tone};
use crate::core::{Rhythm, State};

/// What the column says with nothing playing.
pub const NOTHING: &str = "Nothing playing";

/// And how to play something: the list's keys, from the keymap.
pub fn nothing_hint() -> String {
    let key = |op| keymap::hint(Context::List, op);
    format!(
        "{} queues the selected file; {} plays it too",
        key(Op::Open),
        key(Op::QueuePlay)
    )
}

/// How long the beat pulse stays lit after a beat, in seconds.
pub const PULSE_SECONDS: f64 = 0.12;

/// The right column (or the summary) as it is now.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// The track playing; `None`: nothing plays.
    pub track: Option<Track>,
}

/// The track playing, as the column shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    /// Playing, paused or stopped.
    pub play: PlayState,
    /// Its title (its file's name without one): a link to it in the
    /// browser.
    pub title: String,
    /// Its artist: a link to the artist's folder.
    pub artist: Option<String>,
    /// Its album: a link to the album's folder.
    pub album: Option<String>,
    /// What the file is, then the way it takes to the device.
    pub tags: Vec<Tag>,
    /// Its tempo, and its key once found (`128 BPM · E major (12B)`),
    /// beside the beat pulse; `None` with the tempo off.
    pub tempo: Option<String>,
}

/// A tag of the column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tag {
    /// What it says.
    pub text: String,
    /// What it is.
    pub kind: TagKind,
    /// A step of the way to the device: an arrow comes before it.
    pub step: bool,
}

/// What a tag is, which sets its colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagKind {
    /// What the file is (the status light): PCM, or a lossy file's codec
    /// and bit rate.
    Status,
    /// The file's rate, where the way starts.
    Source,
    /// What to look at: a resampling step, samples clipped.
    Warn,
    /// The device: a click opens its setting.
    Device,
    /// A detail of the way: what the file was decoded from.
    Detail,
}

impl Info {
    /// The column for `state`.
    pub fn of(state: &State) -> Info {
        let Some(track) = &state.playing.track else {
            return Info { track: None };
        };
        let tags = &track.info.tags;
        let (_, status, _) = text::status_light(state);
        let mut shown = vec![Tag {
            text: status,
            kind: TagKind::Status,
            step: false,
        }];
        shown.extend(route(state));
        let tempo = text::bpm_line(state).map(|segs| {
            segs.iter()
                .filter(|s| s.tone != Tone::Pulse)
                .map(|s| s.text.as_str())
                .collect::<String>()
                .trim()
                .to_owned()
        });
        Info {
            track: Some(Track {
                play: state.playing.state,
                title: text::track_title(&track.info),
                artist: tags.artist.clone().filter(|a| !a.is_empty()),
                album: tags.album.clone().filter(|a| !a.is_empty()),
                tags: shown,
                tempo,
            }),
        }
    }

    /// The summary's outline (the column's is [`Info`]'s `Display`): the
    /// play state, title and artist, then the tags.
    pub fn summary(&self) -> String {
        let Some(track) = &self.track else {
            return format!("[summary] {NOTHING}\n");
        };
        let mut first = format!("{} {}", text::play_symbol(track.play), track.title);
        if let Some(artist) = &track.artist {
            first.push_str(" — ");
            first.push_str(artist);
        }
        format!("[summary] {first}\n{}", tags_outline(&track.tags))
    }
}

/// The way from the file to the device ([`text::signal_path`]) as tags:
/// each step after an arrow, what follows the device on its own.
fn route(state: &State) -> Vec<Tag> {
    let mut tags = Vec::new();
    let mut step = false;
    for seg in text::signal_path(state).unwrap_or_default() {
        if seg.text == " → " {
            step = true;
            continue;
        }
        let kind = match seg.tone {
            Tone::Source => TagKind::Source,
            Tone::Warn => TagKind::Warn,
            Tone::Device => TagKind::Device,
            _ => TagKind::Detail,
        };
        tags.push(Tag {
            text: seg.text.trim_start_matches(", ").to_owned(),
            kind,
            step,
        });
        step = false;
    }
    tags
}

/// The tags' outline: the status light, then the way as the signal path
/// says it.
fn tags_outline(tags: &[Tag]) -> String {
    let mut out = String::new();
    let mut way = String::new();
    for tag in tags {
        if tag.kind == TagKind::Status {
            out.push_str(&format!("[status] ○ {}\n", tag.text));
            continue;
        }
        if tag.step {
            way.push_str(" → ");
        } else if !way.is_empty() {
            way.push_str(", ");
        }
        way.push_str(&tag.text);
    }
    if !way.is_empty() {
        out.push_str(&format!("[route] {way}\n"));
    }
    out
}

impl fmt::Display for Info {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Some(track) = &self.track else {
            writeln!(f, "[title] {NOTHING}")?;
            return writeln!(f, "[dim] {}", nothing_hint());
        };
        writeln!(f, "[title link] {}", track.title)?;
        if let Some(artist) = &track.artist {
            writeln!(f, "[artist link] {artist}")?;
        }
        if let Some(album) = &track.album {
            writeln!(f, "[album link] {album}")?;
        }
        f.write_str(&tags_outline(&track.tags))?;
        // In the column, the device's tag opens its setting.
        if let Some(device) = track.tags.iter().find(|t| t.kind == TagKind::Device) {
            writeln!(f, "[device link] {}", device.text)?;
        }
        if let Some(tempo) = &track.tempo {
            writeln!(f, "[tempo] ♩ {tempo}")?;
        }
        Ok(())
    }
}

/// The beat pulse at a moment: how lit, and whether on a bar's first
/// beat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beat {
    /// 1 on the beat, fading to 0 over [`PULSE_SECONDS`].
    pub lit: f32,
    /// The bar's first beat (in the accent).
    pub downbeat: bool,
}

/// The beat pulse `heard` seconds into the playing track: lit for a
/// moment on each beat where the beat is clear, while it plays; `None`
/// between beats, where the beat is unclear or not known, and while
/// nothing plays.
pub fn pulse(state: &State, heard: f64) -> Option<Beat> {
    if state.playing.state != PlayState::Playing {
        return None;
    }
    let path = state
        .playing
        .track
        .as_ref()
        .and_then(|t| state.queue.paths.get(t.index))?;
    let Some(Rhythm::Ready(r)) = state.rhythm.get(path) else {
        return None;
    };
    let a = &r.analysis;
    let i = a.beat_at(heard)?;
    let since = heard - a.beats[i];
    (since < PULSE_SECONDS && a.is_clear_at(heard)).then(|| Beat {
        lit: (1.0 - since / PULSE_SECONDS).clamp(0.0, 1.0) as f32,
        downbeat: a.is_downbeat(i),
    })
}
