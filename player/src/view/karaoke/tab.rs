//! The Karaoke tab's model ([`Tab`]): the song (its title, its artist and
//! the key it is in), the button into the mode (greyed with why until the
//! song can be sung), the sliders with the sync's row under the delay, the
//! game's switches, what is known of the lyrics of the song and the next
//! two, the microphone (its meter, or why it is not open), the echo
//! canceller and the word timing, a few notes on the mode, what the
//! voice's lane says while it has nothing to show, and the keys for the
//! pane's foot; while a sync shows, its panel ([`crate::view::sync`]).
//! The window draws it ([`crate::gui`]'s Karaoke tab) over the pane the
//! cell layout leaves for it.

use std::fmt;

use crate::core::karaoke::{self, ModelState, SongLyrics};
use crate::core::keymap::Op;
use crate::core::lanes::{rhythmed, Block};
use crate::core::sliders::Slider;
use crate::core::text;
use crate::core::{Gpu, State};
use crate::view::sync::{self, Panel};

/// The label of the button into Karaoke mode.
pub const ENTER: &str = "Enter Karaoke Mode";
/// The sync's button, on its row under the delay.
pub const MEASURE: &str = "Measure";

/// What to do about a microphone giving only silence, for this system.
const SILENT_HINT: &str = if cfg!(target_os = "macos") {
    "Allow it in System Settings › Privacy & Security › Microphone (Jodeldiplom, or the terminal it runs in)."
} else {
    "Check the input is not muted in the system's sound settings."
};

/// A key and what it does, for a pane's foot: `enter` "Karaoke mode"
/// (no key: a word of what to do).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hint {
    /// The key (or keys, as `↑ ↓`).
    pub keys: &'static str,
    /// What it does.
    pub what: &'static str,
}

impl Hint {
    /// `keys` doing `what`.
    pub const fn new(keys: &'static str, what: &'static str) -> Hint {
        Hint { keys, what }
    }
}

impl fmt::Display for Hint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.keys.is_empty() {
            f.write_str(self.what)
        } else {
            write!(f, "{} {}", self.keys, self.what)
        }
    }
}

/// The tab's keys, while no sync shows.
const KEYS: [Hint; 4] = [
    Hint::new("enter", "Karaoke mode"),
    Hint::new("↑ ↓", "row"),
    Hint::new("← →", "its value"),
    Hint::new("s", "sync"),
];

/// The button into the mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enter {
    /// What a press does.
    pub op: Op,
    /// What a press with shift held does: in where the song is.
    pub shifted: Op,
    /// Why the song cannot be sung yet (the button greyed, a press
    /// refused with the same said on the status line).
    pub why: Option<&'static str>,
}

/// A slider's row.
#[derive(Clone, Debug, PartialEq)]
pub struct SliderRow {
    /// Which.
    pub slider: Slider,
    /// Its value as shown (`−12 dB`, `150 ms`, `+2`).
    pub value: String,
    /// Its value as a share of its range: where its knob is.
    pub share: f64,
    /// The arrow keys move it.
    pub focused: bool,
    /// The pointer holds it.
    pub held: bool,
    /// It does nothing here (the voice without a GPU for its stems).
    pub dim: bool,
}

/// A row of the controls: a slider, or the sync's under the delay.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    /// A slider.
    Slider(SliderRow),
    /// The sync: its button, and where the delay in force comes from.
    Sync(String),
}

/// One of the game's settings: on or off, a press turns it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Switch {
    /// What a press does.
    pub op: Op,
    /// Its name.
    pub label: &'static str,
    /// Its key on the tab, if it has one.
    pub key: Option<&'static str>,
    /// On.
    pub on: bool,
    /// What it means as it is.
    pub what: &'static str,
}

/// What is known of a song's lyrics: the song playing first, then the
/// next two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LyricsRow {
    /// The song.
    pub name: String,
    /// What is known (nothing yet: empty).
    pub what: String,
    /// Still looked for.
    pub looking: bool,
    /// Heard from the voice by machine learning, no trusted source
    /// having them: said in the warning's colour.
    pub heard: bool,
    /// The song playing (or the mode's first).
    pub playing: bool,
}

/// The microphone.
#[derive(Clone, Debug, PartialEq)]
pub enum Mic {
    /// Being opened.
    Opening,
    /// Open: its name, and how loud the singer is now (the music's echo
    /// taken out), from −60 dB (0) to 0 dB (1).
    Open {
        /// Its name.
        name: String,
        /// The singer's level.
        level: f64,
    },
    /// Open, and only silence comes in: what to check.
    Silent {
        /// Its name.
        name: String,
        /// What to check, for this system.
        hint: &'static str,
    },
    /// It could not be opened, and why.
    Failed(String),
}

impl Mic {
    /// Its key, while it has one: `i` takes the next input.
    pub fn key(&self) -> Option<Hint> {
        match self {
            Mic::Open { .. } => Some(Hint::new("i", "the next one")),
            Mic::Failed(_) => Some(Hint::new("i", "another one")),
            Mic::Opening | Mic::Silent { .. } => None,
        }
    }
}

/// The Karaoke tab as it is now.
#[derive(Clone, Debug, PartialEq)]
pub struct Tab {
    /// The song playing, or the one the mode starts with.
    pub title: String,
    /// Its artist, and the key it is in once found.
    pub artist: String,
    /// The button into the mode.
    pub enter: Enter,
    /// The sliders, the sync's row under the delay.
    pub controls: Vec<Control>,
    /// The game's settings.
    pub switches: Vec<Switch>,
    /// What is known of the lyrics of the song and the next two.
    pub lyrics: Vec<LyricsRow>,
    /// The microphone.
    pub mic: Mic,
    /// What the echo canceller does, while the microphone hears.
    pub echo: Option<String>,
    /// How the words are timed.
    pub words: &'static str,
    /// What the mode does, in a few lines.
    pub notes: Vec<&'static str>,
    /// What the voice's lane says while it has nothing to show.
    pub voice: Option<&'static str>,
    /// The keys, for the pane's foot (the sync's while one shows).
    pub keys: Vec<Hint>,
    /// The sync's panel, while one shows.
    pub sync: Option<Panel>,
}

impl Tab {
    /// The tab for `state`.
    pub fn of(state: &State) -> Tab {
        let (title, mut artist) = match &state.playing.track {
            Some(t) => (
                text::track_title(&t.info),
                t.info.tags.artist.clone().unwrap_or_default(),
            ),
            None => ("Nothing playing".into(), String::new()),
        };
        let key = rhythmed(state, Block::Now).and_then(|r| r.analysis.key);
        if let Some(key) = key {
            if !artist.is_empty() {
                artist.push_str(" · ");
            }
            artist.push_str(&text::key_text(&key));
        }
        let why = karaoke::song_index(state).and_then(|i| karaoke::not_singable(state, i));
        let sync = Panel::of(state);
        Tab {
            title,
            artist,
            enter: Enter {
                op: Op::KaraokeEnter,
                shifted: Op::KaraokeEnterHere,
                why,
            },
            controls: controls(state),
            switches: switches(state),
            lyrics: lyrics(state),
            mic: mic(state),
            echo: (state.karaoke.mic.name.is_some() && !state.karaoke.mic.silent)
                .then(|| echo_text(state)),
            words: match &state.karaoke.model {
                ModelState::Missing => {
                    "not timed: the speech model is not in this build (make puts it there)"
                }
                ModelState::Coming => {
                    "timed once the speech model is fetched (378 MB, when first needed)"
                }
                ModelState::Ready => "timed against the voice by the speech model",
            },
            notes: notes(state),
            voice: super::voice_note(state),
            keys: if sync.is_some() {
                sync::keys(state)
            } else {
                KEYS.to_vec()
            },
            sync,
        }
    }
}

/// The id slider's track is found by in the window.
pub fn track_id(slider: Slider) -> &'static str {
    match slider {
        Slider::Voice => "slider-voice",
        Slider::Delay => "slider-delay",
        Slider::Key => "slider-key",
    }
}

/// The sliders, top to bottom, the sync's row right after the delay's.
fn controls(state: &State) -> Vec<Control> {
    let mut out = Vec::new();
    for slider in Slider::ALL {
        out.push(Control::Slider(SliderRow {
            slider,
            value: slider.text(state).trim().to_owned(),
            share: slider.share(state),
            focused: state.karaoke.focus == slider,
            held: state.karaoke.dragging == Some(slider),
            // The voice's level means nothing without stems.
            dim: slider == Slider::Voice && !matches!(state.gpu, Gpu::Present(_)),
        }));
        if slider == Slider::Delay {
            out.push(Control::Sync(sync::source_text(state)));
        }
    }
    out
}

/// The game on or off, the notes shown or hidden, the singing kept; each
/// a press (or its key: g, p).
fn switches(state: &State) -> Vec<Switch> {
    let c = &state.config;
    vec![
        Switch {
            op: Op::KaraokeGame,
            label: "Game",
            key: Some("g"),
            on: c.karaoke_game,
            what: if c.karaoke_game {
                "your singing scored line by line against the original"
            } else {
                "just singing, nothing scored"
            },
        },
        Switch {
            op: Op::KaraokeNotes,
            label: "Notes",
            key: Some("p"),
            on: c.karaoke_notes,
            what: if c.karaoke_notes {
                "the song's notes run by in the mode, for you to sing on"
            } else {
                "only the words"
            },
        },
        Switch {
            op: Op::KaraokeKeep,
            label: "Keep",
            key: None,
            on: c.karaoke_keep,
            what: if c.karaoke_keep {
                "your singing of each song saved (recordings, beside the settings), to tune the game"
            } else {
                "your singing is not saved"
            },
        },
    ]
}

/// What is known of the lyrics of the song and the next two.
fn lyrics(state: &State) -> Vec<LyricsRow> {
    let Some(first) = karaoke::song_index(state) else {
        return Vec::new();
    };
    (state.queue.paths.iter().enumerate().skip(first).take(3))
        .map(|(k, path)| {
            let known = karaoke::lyrics_of(state, k);
            LyricsRow {
                name: crate::core::shown_title(state, path),
                what: known.map(SongLyrics::describe).unwrap_or_default(),
                looking: matches!(known, Some(SongLyrics::Pending)),
                heard: known.is_some_and(SongLyrics::heard),
                playing: k == first,
            }
        })
        .collect()
}

/// The microphone: which, and how loud the singer is now; or why it is
/// not open.
fn mic(state: &State) -> Mic {
    let mic = &state.karaoke.mic;
    match (&mic.name, &mic.error) {
        (Some(name), _) if mic.silent => Mic::Silent {
            name: name.clone(),
            hint: SILENT_HINT,
        },
        (Some(name), _) => {
            let peak = state.mic.map_or(0.0, |m| m.singer());
            let db = 20.0 * f64::from(peak.max(1e-6)).log10();
            Mic::Open {
                name: name.clone(),
                level: ((db + 60.0) / 60.0).clamp(0.0, 1.0),
            }
        }
        (None, Some(why)) => Mic::Failed(why.clone()),
        (None, None) => Mic::Opening,
    }
}

/// What the mode does, the voice's stems first.
fn notes(state: &State) -> Vec<&'static str> {
    let gpu = match state.gpu {
        Gpu::Present(_) => "Here and in the mode the voice plays at its level, the rest as it is.",
        Gpu::Looking => "Still looking for a GPU for the stems.",
        Gpu::Absent => "No GPU: the voice is turned down only where its stems are kept.",
    };
    vec![
        gpu,
        "Each song waits for START, then counts in 3, 2, 1, a second each; again after a break of 8 s.",
        "The microphone only listens: it is never played; the music stays as it is.",
        "The music's echo is taken out of what it hears; your voice never is.",
        "esc in the mode comes back here.",
    ]
}

/// What the tab says of the echo canceller.
fn echo_text(state: &State) -> String {
    match (state.config.karaoke_echo, state.mic.and_then(|m| m.echo())) {
        (false, _) => "off (karaoke_echo in the settings file)".to_owned(),
        (true, None) => "on, once music plays".to_owned(),
        (true, Some((_, false, _))) => "learning the room…".to_owned(),
        (true, Some((db, true, delay))) => match delay {
            Some(d) => format!(
                "the music's echo taken out, {db:.0} dB (it comes {:.0} ms late)",
                d * 1000.0
            ),
            None => format!("the music's echo taken out, {db:.0} dB"),
        },
    }
}

/// The keys as a line: `enter Karaoke mode · ↑ ↓ row`.
pub fn keys_line(keys: &[Hint]) -> String {
    let keys: Vec<String> = keys.iter().map(Hint::to_string).collect();
    keys.join(" · ")
}

/// An outline of the tab, one line a piece with its tone in brackets:
/// for the snapshots.
impl fmt::Display for Tab {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[title] {}", self.title)?;
        if !self.artist.is_empty() {
            writeln!(f, "[dim] {}", self.artist)?;
        }
        match self.enter.why {
            None => writeln!(f, "[primary] {ENTER} (enter)")?,
            Some(why) => writeln!(f, "[greyed] {ENTER} (enter)  [dim] {why}")?,
        }
        for control in &self.controls {
            match control {
                Control::Slider(s) => {
                    let mut tone = String::from("slider");
                    for (on, word) in [(s.focused, " focused"), (s.held, " held"), (s.dim, " dim")]
                    {
                        if on {
                            tone.push_str(word);
                        }
                    }
                    writeln!(
                        f,
                        "[{tone}] {} {:.0} % {}",
                        s.slider.label(),
                        s.share * 100.0,
                        s.value
                    )?;
                }
                Control::Sync(from) => {
                    writeln!(f, "[row] Sync  [button] {MEASURE} (s)  [dim] {from}")?
                }
            }
        }
        for s in &self.switches {
            let on = if s.on { "on" } else { "off" };
            let key = s.key.map(|k| format!(" ({k})")).unwrap_or_default();
            writeln!(f, "[switch {on}] {}{key}: {}", s.label, s.what)?;
        }
        for l in &self.lyrics {
            let mut tone = String::from("lyrics");
            if l.playing {
                tone.push_str(" playing");
            }
            if l.looking {
                tone.push_str(" looking");
            }
            if l.heard {
                tone.push_str(" heard");
            }
            if l.what.is_empty() {
                writeln!(f, "[{tone}] {}", l.name)?;
            } else {
                writeln!(f, "[{tone}] {}: {}", l.name, l.what)?;
            }
            if l.heard && l.playing {
                writeln!(f, "[warn] {}", karaoke::HEARD_WARNING)?;
            }
        }
        match &self.mic {
            Mic::Opening => writeln!(f, "[mic] opening…")?,
            Mic::Open { name, level } => writeln!(f, "[mic] {name} {:.0} %", level * 100.0)?,
            Mic::Silent { name, hint } => {
                writeln!(f, "[mic fail] {name}: only silence comes in  [dim] {hint}")?
            }
            Mic::Failed(why) => writeln!(f, "[mic fail] not open: {why}")?,
        }
        if let Some(key) = self.mic.key() {
            writeln!(f, "[key] {key}")?;
        }
        if let Some(echo) = &self.echo {
            writeln!(f, "[echo] {echo}")?;
        }
        writeln!(f, "[words] {}", self.words)?;
        for note in &self.notes {
            writeln!(f, "[note] {note}")?;
        }
        if let Some(voice) = self.voice {
            writeln!(f, "[voice] {voice}")?;
        }
        writeln!(f, "[keys] {}", keys_line(&self.keys))?;
        if let Some(panel) = &self.sync {
            write!(f, "{panel}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tab says what the canceller does: off, waiting for music,
    /// learning, or how much of the echo it takes out.
    #[test]
    fn the_tab_says_what_the_canceller_does() {
        let mut state = State::new("/m".into(), crate::config::Config::default());
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

    /// The sync's row comes right after the delay's; the focused slider and
    /// the one held are said; the voice is dim without a GPU.
    #[test]
    fn the_sync_row_is_under_the_delay() {
        let mut state = State::new("/m".into(), crate::config::Config::default());
        state.karaoke.focus = Slider::Delay;
        state.karaoke.dragging = Some(Slider::Key);
        let tab = Tab::of(&state);
        let rows: Vec<String> = tab
            .controls
            .iter()
            .map(|c| match c {
                Control::Slider(s) => {
                    format!(
                        "{}{}{}",
                        s.slider.label(),
                        if s.focused { "*" } else { "" },
                        if s.held { "!" } else { "" }
                    )
                }
                Control::Sync(_) => "sync".into(),
            })
            .collect();
        assert_eq!(rows, ["Voice", "Delay*", "sync", "Key!"]);
        assert!(matches!(&tab.controls[0], Control::Slider(s) if s.dim));
        assert_eq!(
            keys_line(&tab.keys),
            "enter Karaoke mode · ↑ ↓ row · ← → its value · s sync"
        );
    }
}
