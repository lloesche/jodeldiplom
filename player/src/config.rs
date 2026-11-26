//! The optional config file: defaults for the output device (the system's
//! sound settings set the rate it runs at; a file at another rate is
//! resampled to it), and the player's other settings.
//!
//! The file holds `key = value` lines (a small subset of TOML), one per
//! setting:
//!
//! ```toml
//! headroom = 0.5
//! bpm = true
//! beat_methods = "flux, network"
//! skip_mix = "mix"
//! karaoke_effects = "off"
//! data_folder = "D:\\Music data"
//! ```
//!
//! The settings are the rows of [`SETTINGS`], in the order the file is
//! written: each key, the [`Config`] field it sets (whose doc says what it
//! means) and the kind of value it takes.
//!
//! The file is in the settings folder ([`crate::dirs`]).
//!
//! Command-line options override it for one run. Settings changed in the
//! UI are written back to it ([`Config::merge_changes`]).

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use jodel_playback::{Headroom, Settings, TempoMode};
use jodel_rhythm::Method;

use crate::core::transition::Preset;

/// One setting of the file ([`SETTINGS`]): its key, how its value is read
/// into a [`Config`] and written from it, and how a change made in the
/// player is taken over ([`Config::merge_changes`]).
pub struct Setting {
    /// Its key in the file.
    pub key: &'static str,
    /// The note a file shows in a commented line where the setting is left
    /// out; `None`: nothing is shown.
    absent: Option<&'static str>,
    /// Sets it from the text after `=`; `Err`: what is wrong.
    read: fn(&mut Config, &str) -> Result<(), String>,
    /// The text after `=`; `None`: left out.
    write: fn(&Config) -> Option<String>,
    /// Takes it from `now` (the third) into the file's settings (the
    /// first) where `now` differs from `base` (the second).
    take: fn(&mut Config, &Config, &Config),
}

/// How a kind of value is read from the file and written to it.
struct Kind<T> {
    /// The value of the text after `=`; `Err`: what it takes instead.
    read: fn(&str) -> Result<T, String>,
    /// The text after `=`; `None`: the setting is left out.
    write: fn(&T) -> Option<String>,
}

/// [`SETTINGS`] from rows of `"key" => field as KIND`, with `, absent
/// "note"` for one that may be left out.
macro_rules! settings {
    (@note) => { None };
    (@note $note:literal) => { Some($note) };
    ($($key:literal => $($field:ident).+ as $kind:ident $(, absent $note:literal)?;)*) => {
        /// Every setting of the file, one row each, in the order the file
        /// is written. A new setting is a field of [`Config`] and a row
        /// here.
        pub const SETTINGS: &[Setting] = &[$(Setting {
            key: $key,
            absent: settings!(@note $($note)?),
            read: |config, text| {
                config.$($field).+ =
                    ($kind.read)(text).map_err(|takes| format!("{} takes {takes}", $key))?;
                Ok(())
            },
            write: |config| ($kind.write)(&config.$($field).+),
            take: |file, base, now| {
                if now.$($field).+ != base.$($field).+ {
                    file.$($field).+.clone_from(&now.$($field).+);
                }
            },
        }),*];
    };
}

settings! {
    "device" => device as DEVICE, absent "an id from --list-devices; absent: the system default";
    "headroom" => headroom as HEADROOM;
    "bpm" => bpm as FLAG;
    "beat_voting" => beat_voting as FLAG;
    "beat_methods" => beat_methods as METHODS;
    "beat_method" => beat_method as METHOD;
    "stems" => stems as FLAG;
    "stem_switch" => stems_on_bar as STEM_SWITCH;
    "skip_mix" => skip_mix as SKIP_MIX;
    "skip_preset" => skip_preset as SKIP_PRESET;
    "transition_preset" => transition_preset as PRESET;
    "transition_tempo" => transition_tempo as TEMPO;
    "karaoke_voice" => karaoke_voice as PERCENT;
    "karaoke_mic_device" => karaoke_mic_device as DEVICE, absent "an input id; absent: the system's default";
    "karaoke_online" => karaoke_online as FLAG;
    "karaoke_lead" => karaoke_lead as LEAD;
    "karaoke_echo" => karaoke_echo as FLAG;
    "karaoke_game" => karaoke_game as FLAG;
    "karaoke_notes" => karaoke_notes as FLAG;
    "karaoke_keep" => karaoke_keep as FLAG;
    "karaoke_effects" => karaoke_effects.db as LEVEL;
    "karaoke_effects_mute" => karaoke_effects.mute as FLAG;
    "transition_lyrics" => transition_lyrics as FLAG;
    "data_folder" => data_folder as FOLDER, absent "the folder the data folder is made in; absent: the system's place";
}

/// `true` or `false`.
const FLAG: Kind<bool> = Kind {
    read: |text| match text {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("true or false".into()),
    },
    write: |on| Some(on.to_string()),
};

/// A device's id, in quotes; empty or `"default"`: the system's default
/// (`None`, left out).
const DEVICE: Kind<Option<String>> = Kind {
    read: |text| {
        let id = parse_string(text).ok_or("a string")?;
        Ok((!id.is_empty() && id != "default").then_some(id))
    },
    write: |id| id.as_deref().map(quote),
};

/// dB below full scale ([`Headroom::parse`]).
const HEADROOM: Kind<Headroom> = Kind {
    read: |text| Headroom::parse(&word(text)).ok_or_else(|| "dB: 0, 0.5, 1, 2 or 3".to_owned()),
    write: |headroom| Some(headroom.value().to_string()),
};

/// Beat methods between commas, at least one; kept in [`Method::ALL`]
/// order, each once.
const METHODS: Kind<Vec<Method>> = Kind {
    read: |text| {
        let mut methods = word(text)
            .split(',')
            .map(|id| Method::parse(id.trim()))
            .collect::<Option<Vec<_>>>()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| format!("{}, at least one, between commas", method_ids()))?;
        methods.sort_unstable();
        methods.dedup();
        Ok(methods)
    },
    write: |methods| {
        let ids: Vec<&str> = methods.iter().map(|m| m.id()).collect();
        Some(quote(&ids.join(", ")))
    },
};

/// One beat method.
const METHOD: Kind<Method> = Kind {
    read: |text| Method::parse(&word(text)).ok_or_else(method_ids),
    write: |method| Some(quote(method.id())),
};

/// A value named by one of these words (in quotes or bare), written in
/// quotes.
macro_rules! words {
    ($($word:literal => $value:expr),+ $(,)?) => {
        Kind {
            read: |text| match word(text).as_str() {
                $($word => Ok($value),)+
                _ => Err(one_of([$($word),+])),
            },
            write: |value| {
                [$(($word, $value)),+]
                    .into_iter()
                    .find(|(_, v)| v == value)
                    .map(|(w, _)| quote(w))
            },
        }
    };
}

/// Stems switch on the next bar, or at once.
const STEM_SWITCH: Kind<bool> = words!("bar" => true, "instant" => false);

/// The next track key cuts to it, or mixes into it.
const SKIP_MIX: Kind<bool> = words!("cut" => false, "mix" => true);

/// How a transition's tempi meet.
const TEMPO: Kind<TempoMode> = words!(
    "varispeed" => TempoMode::Varispeed,
    "stretch" => TempoMode::Stretch,
);

/// A transition preset ([`Preset::id`]).
const PRESET: Kind<Preset> = Kind {
    read: |text| Preset::from_id(&word(text)).ok_or_else(|| one_of(Preset::ALL.map(Preset::id))),
    write: |preset| Some(quote(preset.id())),
};

/// The Transition tab's preset as it is set then (`"transition-tab"`,
/// `None`), or a preset of its own.
const SKIP_PRESET: Kind<Option<Preset>> = Kind {
    read: |text| match word(text).as_str() {
        "transition-tab" => Ok(None),
        id => Preset::from_id(id).map(Some).ok_or_else(|| {
            one_of(std::iter::once("transition-tab").chain(Preset::ALL.map(Preset::id)))
        }),
    },
    write: |by| Some(quote(by.map_or("transition-tab", Preset::id))),
};

/// A percentage, 0 to 100.
const PERCENT: Kind<u8> = Kind {
    read: |text| {
        text.parse()
            .ok()
            .filter(|&v| v <= 100)
            .ok_or_else(|| "a percentage, 0 to 100".to_owned())
    },
    write: |percent| Some(percent.to_string()),
};

/// Milliseconds, 0 to [`LEAD_MOST`].
const LEAD: Kind<u16> = Kind {
    read: |text| {
        text.parse()
            .ok()
            .filter(|&v| v <= LEAD_MOST)
            .ok_or_else(|| format!("milliseconds, 0 to {LEAD_MOST}"))
    },
    write: |ms| Some(ms.to_string()),
};

/// A fader's level in dB, −100 to 6, to a tenth of a dB; `"off"`: −∞.
const LEVEL: Kind<f64> = Kind {
    read: |text| match word(text).as_str() {
        "off" => Ok(f64::NEG_INFINITY),
        db => db
            .parse::<f64>()
            .ok()
            .filter(|d| (-100.0..=6.0).contains(d))
            .map(|d| (d * 10.0).round() / 10.0)
            .ok_or_else(|| "a level in dB, -100 to 6, or \"off\"".to_owned()),
    },
    write: |&db| {
        Some(if db > f64::NEG_INFINITY {
            format!("{db:.1}")
        } else {
            quote("off")
        })
    },
};

/// A folder's full path, in quotes (`None`, left out: none set).
const FOLDER: Kind<Option<PathBuf>> = Kind {
    read: |text| {
        parse_string(text)
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .map(Some)
            .ok_or_else(|| "a folder's full path, in quotes".to_owned())
    },
    write: |path| path.as_ref().map(|p| quote(&p.display().to_string())),
};

/// The player's settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The output device id; `None`: the system default.
    pub device: Option<String>,
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
    /// Karaoke: the song's notes show under the line being sung.
    pub karaoke_notes: bool,
    /// Karaoke: each song's singing is kept ([`crate::keep`]).
    pub karaoke_keep: bool,
    /// Karaoke: the game's sound effects ([`crate::sfx`]): the Effects
    /// strip's fader (to a tenth of a dB, −∞ off) and mute (its solo is
    /// not kept).
    pub karaoke_effects: crate::core::desk::StripSetting,
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
            karaoke_notes: true,
            karaoke_keep: false,
            karaoke_effects: crate::core::desk::StripSetting::default(),
            transition_lyrics: true,
            data_folder: None,
        }
    }
}

impl Config {
    /// The engine's rate policy settings.
    pub fn settings(&self) -> Settings {
        Settings {
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
            match SETTINGS.iter().find(|s| s.key == key) {
                Some(setting) => (setting.read)(&mut config, value).map_err(|e| at(&e))?,
                None => warnings.push(at(&format!("unknown setting {key:?}"))),
            }
        }
        Ok((config, warnings))
    }

    /// The file's text for these settings: the rows of [`SETTINGS`] in
    /// order (one left out as its note, commented, where it has one).
    pub fn to_text(&self) -> String {
        let mut text = String::from("# jodeldiplom settings\n");
        for setting in SETTINGS {
            match ((setting.write)(self), setting.absent) {
                (Some(value), _) => {
                    let _ = writeln!(text, "{} = {value}", setting.key);
                }
                (None, Some(note)) => {
                    let _ = writeln!(text, "# {} = \"...\"   ({note})", setting.key);
                }
                (None, None) => {}
            }
        }
        text
    }

    /// The file's settings (`self`) with what was changed in the player
    /// taken over: each setting where `now` differs from `base` (what the
    /// player started with, the command line's overrides in it) is
    /// `now`'s. The overrides themselves stay out of the file.
    pub fn merge_changes(&self, base: &Config, now: &Config) -> Config {
        let mut file = self.clone();
        for setting in SETTINGS {
            (setting.take)(&mut file, base, now);
        }
        file
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

/// The beat methods' ids, as [`one_of`] says them.
fn method_ids() -> String {
    one_of(Method::ALL.iter().map(|m| m.id()))
}

/// Ids in quotes, the last after "or": `"a", "b" or "c"`.
fn one_of<'a>(ids: impl IntoIterator<Item = &'a str>) -> String {
    let ids: Vec<String> = ids.into_iter().map(quote).collect();
    match ids.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// A value in quotes, or bare.
fn word(value: &str) -> String {
    parse_string(value).unwrap_or_else(|| value.to_owned())
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

    /// Every setting away from its default (its fields named one by one,
    /// so a new one cannot be left out).
    fn every_setting_changed() -> Config {
        Config {
            device: Some(r#"{0.0.0}.{a"b\c}"#.into()),
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
            karaoke_notes: false,
            karaoke_keep: true,
            karaoke_effects: crate::core::desk::StripSetting {
                db: -12.5,
                mute: true,
                solo: false,
            },
            transition_lyrics: false,
            data_folder: Some(if cfg!(windows) {
                r"D:\Music data".into()
            } else {
                "/srv/music data".into()
            }),
        }
    }

    #[test]
    fn settings_round_trip() {
        let config = every_setting_changed();
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

    /// Each key is one row's.
    #[test]
    fn every_key_has_one_row() {
        let mut keys: Vec<&str> = SETTINGS.iter().map(|s| s.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), SETTINGS.len());
    }

    /// Only what was changed in the player reaches the file; an override
    /// from the command line does not, and the file's own values stay.
    #[test]
    fn changes_made_in_the_player_are_merged_into_the_file() {
        let (file, _) = Config::parse("bpm = true\nkaraoke_lead = 300\n").unwrap();
        let base = Config {
            headroom: Headroom(10),
            ..file.clone()
        };
        let mut now = base.clone();
        now.bpm = false;
        now.karaoke_effects.db = -6.0;
        let merged = file.merge_changes(&base, &now);
        assert!(!merged.bpm);
        assert_eq!(merged.karaoke_effects.db, -6.0);
        assert!(!merged.karaoke_effects.mute);
        assert_eq!(merged.headroom, Headroom::NONE);
        assert_eq!(merged.karaoke_lead, 300);
        // Every setting is taken over where it changed.
        let every = every_setting_changed();
        let default = Config::default();
        assert_eq!(default.merge_changes(&default, &every), every);
        assert_eq!(every.merge_changes(&every, &default), default);
    }

    #[test]
    fn comments_unknown_keys_and_errors() {
        let (config, warnings) = Config::parse(
            "# hi\nbpm = false # trailing\nvolume = 3\nrate.\"default\" = 44100\n[section]\nresample = false\n",
        )
        .unwrap();
        assert!(!config.bpm);
        // A device's rate was a setting once, and resampling could be off;
        // the system sets the rate now, and a file at another is resampled.
        assert_eq!(
            warnings,
            vec![
                "line 3: unknown setting \"volume\"",
                "line 4: unknown setting \"rate.\\\"default\\\"\"",
                "line 6: unknown setting \"resample\""
            ]
        );
        assert!(!config.to_text().contains("rate."));
        assert!(!config.to_text().contains("resample"));
        assert!(Config::parse("bpm = maybe").is_err());
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
