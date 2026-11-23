//! The optional config file: defaults for the output device, the rate cap
//! and whether to resample when no rate carries a file bit-perfect, and
//! the player's other settings.
//!
//! The file holds `key = value` lines (a small subset of TOML):
//!
//! ```toml
//! device = "{0.0.0.00000000}.{...}"   # an id from --list-devices; absent: the system default
//! max_rate = 192000                   # absent or 0: no cap
//! resample = true                     # false: skip files no rate carries bit-perfect
//! headroom = 0                        # dB below full scale where not bit-perfect: 0, 0.5, 1, 2, 3 (lossy files at least 1.5)
//! bpm = true                          # false: no tempo, no beat pulse, no beat analysis
//! beat_voting = true                  # true: the beat_methods vote; false: beat_method decides
//! beat_methods = "flux, network"      # the methods that vote (at least one)
//! beat_method = "network"             # the one method without voting
//! stems = true                        # true: the cues listen to stems separated on the GPU
//! stem_switch = "bar"                 # stems switch on the next bar ("bar") or at once ("instant")
//! skip_mix = "cut"                    # the next track key: "cut", or "mix" into it
//! skip_preset = "transition-tab"      # the preset it mixes by: the Transition tab's, or one of its own
//! transition_preset = "fade"          # the Transition tab's transition preset, as last chosen
//! transition_tempo = "varispeed"      # how a transition's tempi meet: "varispeed" or "stretch"
//!                                     # ("fade", "drums-first", "bass-swap", "vocals-last")
//! karaoke_voice = 0                   # karaoke mode: the voice stem's slider, 0 to 100 % of its travel (100: 0 dB)
//! karaoke_mic_device = "..."          # the microphone (an input id); absent: the system's default
//! karaoke_online = true               # karaoke: look lyrics up on LRCLIB (false: tags and .lrc files only)
//! karaoke_lead = 200                  # karaoke: the words light up this many ms before they are sung (0 to 1000)
//! karaoke_echo = true                 # karaoke: take the music's echo (from the speakers) out of the singer the microphone hears
//! karaoke_game = true                 # karaoke: score the singing (points, medals, high scores); false: just sing
//! karaoke_level = "medium"            # karaoke: how strictly it is scored: "easy" (any key, the shape), "medium" (any key, the steps), "hard" (the song's key)
//! karaoke_notes = true                # karaoke: the song's notes under the line being sung
//! karaoke_keep = false                # karaoke: keep your singing of each song (recordings, beside this file), to tune the game
//! transition_lyrics = true            # the Transition tab: the lyrics on the vocals lane (L)
//! data_folder = "D:\\Music data"      # the folder the data folder is made in; absent: the system's place
//! ```
//!
//! The file is in the settings folder ([`crate::dirs`]).
//!
//! Command-line options override it for one run. Settings changed in the
//! UI are written back to it.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use jodel_playback::{Headroom, Settings};
use jodel_rhythm::Method;

use crate::core::transition::Preset;

/// The player's settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The output device id; `None`: the system default.
    pub device: Option<String>,
    /// The highest output rate, if capped.
    pub max_rate: Option<u32>,
    /// Resample when no rate the device takes carries a file bit-perfect
    /// (`false`: such files are skipped).
    pub resample: bool,
    /// The gain below full scale on resampled routes.
    pub headroom: Headroom,
    /// Analyse the beats of the playing tracks and show the tempo and a
    /// beat pulse.
    pub bpm: bool,
    /// The beat methods vote ([`Config::beat_methods`]); otherwise one
    /// decides ([`Config::beat_method`]).
    pub beat_voting: bool,
    /// The methods that vote, in [`Method::ALL`] order; never empty.
    pub beat_methods: Vec<Method>,
    /// The method that decides without voting.
    pub beat_method: Method,
    /// Separate the music into stems on the GPU for the beat analysis's
    /// cues, when there is a GPU (else the fast split of the mix).
    pub stems: bool,
    /// What the next track key does: cut to it (`false`), or mix into it,
    /// beat-matched, by [`Config::skip_preset`].
    pub skip_mix: bool,
    /// The preset the next track key mixes by: `None`, the Transition tab's as
    /// it is set then (its fades as edited too); else one of its own.
    pub skip_preset: Option<Preset>,
    /// The Transition tab's transition preset, as last chosen (`p`).
    pub transition_preset: Preset,
    /// How a transition's tempi meet, as last chosen.
    pub transition_tempo: jodel_playback::TempoMode,
    /// The Transition tab shows the lyrics on the vocals lane (`L`).
    pub transition_lyrics: bool,
    /// A stem turned on or off is heard from the next bar's first beat
    /// (where the playing track's bars are known); else as soon as the
    /// output has played what it holds.
    pub stems_on_bar: bool,
    /// Karaoke mode: the voice stem's level, as a position along its
    /// slider in percent (0, −∞, to 100, 0 dB), read on the fader's taper
    /// ([`crate::core::sliders::voice_db`]).
    pub karaoke_voice: u8,
    /// The microphone, an input device id; `None`: the system's default.
    pub karaoke_mic_device: Option<String>,
    /// Karaoke: look lyrics up on the web (LRCLIB) when the track has
    /// none in its tags or beside it.
    pub karaoke_online: bool,
    /// Karaoke: how far ahead of the voice the words light up, in ms (0 to
    /// [`LEAD_MOST`]), so they can be read before they are sung.
    pub karaoke_lead: u16,
    /// Karaoke: the echo canceller takes what plays (heard again from the
    /// speakers) out of the singer the microphone hears. The microphone is
    /// never played either way.
    pub karaoke_echo: bool,
    /// Karaoke: the singing is scored (points, medals, high scores).
    pub karaoke_game: bool,
    /// Karaoke: how strictly the singing is scored.
    pub karaoke_level: crate::core::singing::Level,
    /// Karaoke: the song's notes show under the line being sung.
    pub karaoke_notes: bool,
    /// Karaoke: each song's singing is kept ([`crate::keep`]).
    pub karaoke_keep: bool,
    /// The folder the data folder ([`crate::dirs::DATA`]) is in; `None`:
    /// the system's place for it ([`crate::dirs::default_data_parent`]).
    pub data_folder: Option<PathBuf>,
}

/// The most [`Config::karaoke_lead`] takes, in ms.
pub const LEAD_MOST: u16 = 1000;

impl Default for Config {
    fn default() -> Self {
        Config {
            device: None,
            max_rate: None,
            resample: true,
            headroom: Headroom::NONE,
            bpm: true,
            beat_voting: true,
            beat_methods: Method::ALL.to_vec(),
            beat_method: Method::Network,
            stems: true,
            skip_mix: false,
            skip_preset: None,
            transition_preset: Preset::default(),
            transition_tempo: jodel_playback::TempoMode::default(),
            stems_on_bar: true,
            karaoke_voice: 0,
            karaoke_mic_device: None,
            karaoke_online: true,
            karaoke_lead: 200,
            karaoke_echo: true,
            karaoke_game: true,
            karaoke_level: crate::core::singing::Level::Medium,
            karaoke_notes: true,
            karaoke_keep: false,
            transition_lyrics: true,
            data_folder: None,
        }
    }
}

/// The rate caps the settings screen offers, highest first (`None`: no cap).
pub const RATE_CAPS: [Option<u32>; 9] = [
    None,
    Some(384_000),
    Some(352_800),
    Some(192_000),
    Some(176_400),
    Some(96_000),
    Some(88_200),
    Some(48_000),
    Some(44_100),
];

impl Config {
    /// The engine's rate policy settings.
    pub fn settings(&self) -> Settings {
        Settings {
            max_rate: self.max_rate,
            resample: self.resample,
            headroom: self.headroom,
        }
    }

    /// The beat methods in force: the voting ones, or the one.
    pub fn methods_in_force(&self) -> Vec<Method> {
        if self.beat_voting {
            self.beat_methods.clone()
        } else {
            vec![self.beat_method]
        }
    }

    /// Parses a config file's text. Unknown keys are reported, not fatal.
    pub fn parse(text: &str) -> Result<(Config, Vec<String>), String> {
        let mut config = Config::default();
        let mut warnings = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let line = strip_comment(line).trim();
            if line.is_empty() || line.starts_with('[') {
                continue;
            }
            let at = |what: &str| format!("line {}: {what}", number + 1);
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| at("expected key = value"))?;
            let (key, value) = (key.trim(), value.trim());
            match key {
                "device" => {
                    let id = parse_string(value).ok_or_else(|| at("device takes a string"))?;
                    config.device = (!id.is_empty() && id != "default").then_some(id);
                }
                "max_rate" => {
                    let rate: u32 = value
                        .parse()
                        .map_err(|_| at("max_rate takes a rate in Hz"))?;
                    config.max_rate = (rate > 0).then_some(rate);
                }
                "resample" => {
                    config.resample = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("resample takes true or false")),
                    }
                }
                "headroom" => {
                    let text = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.headroom = Headroom::parse(&text)
                        .ok_or_else(|| at("headroom takes dB: 0, 0.5, 1, 2 or 3"))?;
                }
                "bpm" => {
                    config.bpm = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("bpm takes true or false")),
                    }
                }
                "stems" => {
                    config.stems = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("stems takes true or false")),
                    }
                }
                "beat_voting" => {
                    config.beat_voting = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("beat_voting takes true or false")),
                    }
                }
                "beat_methods" => {
                    let text = parse_string(value).unwrap_or_else(|| value.to_owned());
                    let mut methods = text
                        .split(',')
                        .map(|id| Method::parse(id.trim()))
                        .collect::<Option<Vec<_>>>()
                        .filter(|m| !m.is_empty())
                        .ok_or_else(|| at(&methods_hint("beat_methods takes")))?;
                    methods.sort_unstable();
                    methods.dedup();
                    config.beat_methods = methods;
                }
                "beat_method" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.beat_method = Method::parse(&id)
                        .ok_or_else(|| at(&methods_hint("beat_method takes one of")))?;
                }
                "stem_switch" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.stems_on_bar = match id.as_str() {
                        "bar" => true,
                        "instant" => false,
                        _ => return Err(at("stem_switch takes \"bar\" or \"instant\"")),
                    };
                }
                "skip_mix" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.skip_mix = match id.as_str() {
                        "cut" => false,
                        "mix" => true,
                        _ => return Err(at("skip_mix takes \"cut\" or \"mix\"")),
                    };
                }
                "skip_preset" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.skip_preset = match id.as_str() {
                        "transition-tab" => None,
                        id => Some(Preset::from_id(id).ok_or_else(|| {
                            at("skip_preset takes \"transition-tab\", \"fade\", \"drums-first\", \"bass-swap\" or \"vocals-last\"")
                        })?),
                    };
                }
                "transition_preset" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.transition_preset = Preset::from_id(&id).ok_or_else(|| {
                        at("transition_preset takes \"fade\", \"drums-first\", \"bass-swap\" or \"vocals-last\"")
                    })?;
                }
                "transition_tempo" => {
                    let id = parse_string(value).unwrap_or_else(|| value.to_owned());
                    config.transition_tempo = match id.as_str() {
                        "varispeed" => jodel_playback::TempoMode::Varispeed,
                        "stretch" => jodel_playback::TempoMode::Stretch,
                        _ => return Err(at("transition_tempo takes \"varispeed\" or \"stretch\"")),
                    };
                }
                "karaoke_voice" => {
                    config.karaoke_voice = value
                        .parse::<u8>()
                        .ok()
                        .filter(|&v| v <= 100)
                        .ok_or_else(|| at("karaoke_voice takes a percentage, 0 to 100"))?;
                }
                "karaoke_mic_device" => {
                    let id = parse_string(value)
                        .ok_or_else(|| at("karaoke_mic_device takes a string"))?;
                    config.karaoke_mic_device = (!id.is_empty() && id != "default").then_some(id);
                }
                // An older file's microphone level: the microphone is no
                // longer played, so it has none.
                "karaoke_mic" => {}
                "karaoke_online" => {
                    config.karaoke_online = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("karaoke_online takes true or false")),
                    }
                }
                "transition_lyrics" => {
                    config.transition_lyrics = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("transition_lyrics takes true or false")),
                    }
                }
                "karaoke_echo" => {
                    config.karaoke_echo = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("karaoke_echo takes true or false")),
                    }
                }
                "data_folder" => {
                    let path = parse_string(value)
                        .map(PathBuf::from)
                        .filter(|p| p.is_absolute())
                        .ok_or_else(|| at("data_folder takes a folder's full path, in quotes"))?;
                    config.data_folder = Some(path);
                }
                "karaoke_game" => {
                    config.karaoke_game = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("karaoke_game takes true or false")),
                    }
                }
                "karaoke_notes" => {
                    config.karaoke_notes = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("karaoke_notes takes true or false")),
                    }
                }
                "karaoke_keep" => {
                    config.karaoke_keep = match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(at("karaoke_keep takes true or false")),
                    }
                }
                "karaoke_level" => {
                    config.karaoke_level = parse_string(value)
                        .as_deref()
                        .and_then(crate::core::singing::Level::parse)
                        .ok_or_else(|| {
                            at("karaoke_level takes \"easy\", \"medium\" or \"hard\"")
                        })?;
                }
                "karaoke_lead" => {
                    config.karaoke_lead = value
                        .parse::<u16>()
                        .ok()
                        .filter(|&v| v <= LEAD_MOST)
                        .ok_or_else(|| at("karaoke_lead takes milliseconds, 0 to 1000"))?;
                }
                other => warnings.push(at(&format!("unknown setting {other:?}"))),
            }
        }
        Ok((config, warnings))
    }

    /// The file's text for these settings.
    pub fn to_text(&self) -> String {
        let mut text = String::from("# jodeldiplom settings\n");
        match &self.device {
            Some(id) => {
                let _ = writeln!(text, "device = {}", quote(id));
            }
            None => text.push_str(
                "# device = \"...\"   (an id from --list-devices; absent: the system default)\n",
            ),
        }
        let _ = writeln!(text, "max_rate = {}", self.max_rate.unwrap_or(0));
        let _ = writeln!(text, "resample = {}", self.resample);
        let _ = writeln!(text, "headroom = {}", self.headroom.value());
        let _ = writeln!(text, "bpm = {}", self.bpm);
        let _ = writeln!(text, "beat_voting = {}", self.beat_voting);
        let ids: Vec<&str> = self.beat_methods.iter().map(|m| m.id()).collect();
        let _ = writeln!(text, "beat_methods = {}", quote(&ids.join(", ")));
        let _ = writeln!(text, "beat_method = {}", quote(self.beat_method.id()));
        let _ = writeln!(text, "stems = {}", self.stems);
        let switch = if self.stems_on_bar { "bar" } else { "instant" };
        let _ = writeln!(text, "stem_switch = {}", quote(switch));
        let skip = if self.skip_mix { "mix" } else { "cut" };
        let _ = writeln!(text, "skip_mix = {}", quote(skip));
        let by = self.skip_preset.map_or("transition-tab", Preset::id);
        let _ = writeln!(text, "skip_preset = {}", quote(by));
        let _ = writeln!(
            text,
            "transition_preset = {}",
            quote(self.transition_preset.id())
        );
        let tempo = match self.transition_tempo {
            jodel_playback::TempoMode::Varispeed => "varispeed",
            jodel_playback::TempoMode::Stretch => "stretch",
        };
        let _ = writeln!(text, "transition_tempo = {}", quote(tempo));
        let _ = writeln!(text, "karaoke_voice = {}", self.karaoke_voice);
        if let Some(id) = &self.karaoke_mic_device {
            let _ = writeln!(text, "karaoke_mic_device = {}", quote(id));
        }
        let _ = writeln!(text, "karaoke_online = {}", self.karaoke_online);
        let _ = writeln!(text, "karaoke_lead = {}", self.karaoke_lead);
        let _ = writeln!(text, "karaoke_echo = {}", self.karaoke_echo);
        let _ = writeln!(text, "karaoke_game = {}", self.karaoke_game);
        let _ = writeln!(text, "karaoke_level = {}", quote(self.karaoke_level.name()));
        let _ = writeln!(text, "karaoke_notes = {}", self.karaoke_notes);
        let _ = writeln!(text, "karaoke_keep = {}", self.karaoke_keep);
        let _ = writeln!(text, "transition_lyrics = {}", self.transition_lyrics);
        match &self.data_folder {
            Some(path) => {
                let _ = writeln!(text, "data_folder = {}", quote(&path.display().to_string()));
            }
            None => text.push_str(
                "# data_folder = \"...\"   (the folder the data folder is made in; absent: the system's place)\n",
            ),
        }
        text
    }

    /// Reads `path`; a missing file gives the defaults.
    pub fn load(path: &Path) -> Result<(Config, Vec<String>), String> {
        match fs::read_to_string(path) {
            Ok(text) => Config::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok((Config::default(), Vec::new())),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Writes the settings to `path`, creating its folder.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        // Whole or not at all: a crash mid-save leaves the settings as they
        // were.
        durable::write_atomic(path, self.to_text().as_bytes())
    }
}

/// `what` and the method names.
fn methods_hint(what: &str) -> String {
    let ids: Vec<String> = Method::ALL
        .iter()
        .map(|m| format!("{:?}", m.id()))
        .collect();
    format!("{what} {}", ids.join(", "))
}

fn strip_comment(line: &str) -> &str {
    let (mut quoted, mut escaped) = (false, false);
    for (i, c) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_string(value: &str) -> Option<String> {
    let inner = value.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn quote(text: &str) -> String {
    let mut out = String::from('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Where the config file lives when `--config` is not given: in the
/// settings folder ([`crate::dirs::settings_dir`]).
pub fn default_path() -> Option<PathBuf> {
    crate::dirs::settings_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let config = Config {
            device: Some(r#"{0.0.0}.{a"b\c}"#.into()),
            max_rate: Some(192_000),
            resample: false,
            headroom: Headroom(5),
            bpm: false,
            beat_voting: false,
            beat_methods: vec![Method::Network],
            beat_method: Method::Flux,
            stems: false,
            skip_mix: true,
            skip_preset: Some(crate::core::transition::Preset::DrumsFirst),
            transition_preset: crate::core::transition::Preset::BassSwap,
            transition_tempo: jodel_playback::TempoMode::Stretch,
            stems_on_bar: false,
            karaoke_voice: 30,
            karaoke_mic_device: Some("{0.0.1.00000000}.{mic}".into()),
            karaoke_online: false,
            karaoke_lead: 350,
            karaoke_echo: false,
            karaoke_game: false,
            karaoke_level: crate::core::singing::Level::Hard,
            karaoke_notes: false,
            karaoke_keep: true,
            transition_lyrics: false,
            data_folder: Some(if cfg!(windows) {
                r"D:\Music data".into()
            } else {
                "/srv/music data".into()
            }),
        };
        let (back, warnings) = Config::parse(&config.to_text()).unwrap();
        assert_eq!(back, config);
        assert!(warnings.is_empty());
        let (default, _) = Config::parse(&Config::default().to_text()).unwrap();
        assert_eq!(default, Config::default());
    }

    /// The next track key's setting is "cut" or "mix", nothing else.
    #[test]
    fn skip_mix_takes_cut_or_mix() {
        let (config, warnings) = Config::parse("skip_mix = \"mix\"\n").unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(config.skip_mix);
        assert!(config.to_text().contains("skip_mix = \"mix\""));
        assert!(!Config::parse("skip_mix = \"cut\"\n").unwrap().0.skip_mix);
        assert!(Config::parse("skip_mix = \"fade\"\n").is_err());
        assert!(Config::parse("skip_mix = \"sometimes\"\n").is_err());
        assert!(Config::parse("skip_preset = \"nope\"\n").is_err());
    }

    /// An older file's microphone level (the microphone is no longer
    /// played, so it has none) is read without a word and not written back.
    #[test]
    fn an_older_microphone_level_is_passed_over() {
        let (config, warnings) = Config::parse("karaoke_mic = 150\nkaraoke_voice = 20\n").unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(config.karaoke_voice, 20);
        assert!(!config.to_text().contains("karaoke_mic ="));
    }

    #[test]
    fn comments_unknown_keys_and_errors() {
        let (config, warnings) = Config::parse(
            "# hi\nresample = false # trailing\nvolume = 3\nmax_rate = 0\n[section]\n",
        )
        .unwrap();
        assert!(!config.resample);
        assert_eq!(config.max_rate, None);
        assert_eq!(warnings, vec!["line 3: unknown setting \"volume\""]);
        assert!(Config::parse("max_rate = fast").is_err());
        assert!(Config::parse("resample = maybe").is_err());
        assert!(!Config::parse("resample = false").unwrap().0.resample);
        assert!(Config::parse("just words").is_err());
    }

    #[test]
    fn headroom_round_trips() {
        for headroom in Headroom::CHOICES {
            let config = Config {
                headroom,
                ..Config::default()
            };
            let text = config.to_text();
            assert!(text.contains(&format!("headroom = {}\n", headroom.value())));
            assert_eq!(Config::parse(&text).unwrap().0, config);
        }
        assert_eq!(
            Config::parse("headroom = \"1\"").unwrap().0.headroom,
            Headroom(10)
        );
        assert_eq!(Config::parse("").unwrap().0.headroom, Headroom::NONE);
        assert!(Config::parse("headroom = loud").is_err());
    }

    #[test]
    fn beat_methods() {
        let (c, _) = Config::parse("beat_methods = \"network, flux, flux\"").unwrap();
        assert_eq!(c.beat_methods, vec![Method::Flux, Method::Network]);
        let (c, _) = Config::parse("beat_methods = flux\nbeat_method = flux").unwrap();
        assert_eq!(c.beat_methods, vec![Method::Flux]);
        assert_eq!(c.beat_method, Method::Flux);
        assert!(Config::parse("beat_methods = \"\"").is_err());
        assert!(Config::parse("beat_methods = \"flux, magic\"").is_err());
        assert!(Config::parse("beat_method = magic").is_err());
        assert!(Config::parse("beat_voting = maybe").is_err());
        let mut c = Config::default();
        assert_eq!(c.methods_in_force(), Method::ALL.to_vec());
        c.beat_voting = false;
        assert_eq!(c.methods_in_force(), vec![Method::Network]);
    }

    #[test]
    fn the_karaoke_online_switch() {
        assert!(Config::default().karaoke_online);
        assert!(
            !Config::parse("karaoke_online = false")
                .unwrap()
                .0
                .karaoke_online
        );
        assert!(
            Config::parse("karaoke_online = true")
                .unwrap()
                .0
                .karaoke_online
        );
        assert!(Config::parse("karaoke_online = maybe").is_err());
    }

    #[test]
    fn the_karaoke_lead() {
        assert_eq!(Config::default().karaoke_lead, 200);
        assert_eq!(Config::parse("karaoke_lead = 0").unwrap().0.karaoke_lead, 0);
        assert!(Config::parse("karaoke_lead = 1001").is_err());
        assert!(Config::parse("karaoke_lead = soon").is_err());
    }

    #[test]
    fn a_missing_file_gives_defaults() {
        let path = std::env::temp_dir().join("jodeldiplom-no-such-dir/none.toml");
        assert_eq!(Config::load(&path).unwrap().0, Config::default());
    }
}
