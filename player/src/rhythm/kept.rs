//! The analyses kept on disk: one small text file a track and recipe
//! (its beats, bars, key and the rest: [`to_text`]) and its waveform
//! beside it, named by a hash of the file as the caches know it
//! ([`identity`]), [`jodel_rhythm::VERSION`] and the recipe ([`cache_key`]).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use jodel_rhythm::meter::{Grouping, Span};
use jodel_rhythm::waveform::Waveform;
use jodel_rhythm::{Analysis, CandidateReport, Method, Region, VERSION};

use crate::cache;
use crate::core::RhythmRecipe;

/// Where analyses are kept: `analyses` in the data folder of the usual
/// settings ([`crate::dirs::Data::current`]; the player itself is told its
/// own).
pub fn cache_dir() -> Option<PathBuf> {
    crate::dirs::Data::current().map(|d| d.analyses())
}

/// What identifies a file's analysis, `IDENTITY vVERSION METHODS`: the
/// file ([`cache::identity`]), the analysis version, the methods (and
/// `+stems` where the cues listened to stems).
pub fn cache_key(path: &Path, methods: &[Method], stems: bool) -> Option<String> {
    let ids: Vec<&str> = methods.iter().map(|m| m.id()).collect();
    let stems = if stems { "+stems" } else { "" };
    Some(format!(
        "{} v{VERSION} {}{stems}",
        cache::identity(path)?,
        ids.join("+")
    ))
}

/// The file and the analysis version a key names ([`cache_key`]): its
/// identity ([`cache::identity_parts`] reads it) and the version.
pub fn key_parts(key: &str) -> Option<(&str, u32)> {
    let mut parts = key.rsplitn(3, ' ');
    let _methods = parts.next()?;
    let version = parts.next()?.strip_prefix('v')?.parse().ok()?;
    Some((parts.next()?, version))
}

/// The cache file for `key` in `dir`.
fn cache_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{}.txt", cache::name(key)))
}

/// Beat times are stored in units of 0.1 ms.
const TICKS: f64 = 10_000.0;

/// Appends `beats` to `s` as ` d1 d2 ...`: each the ticks since the last.
pub(crate) fn beats_text(s: &mut String, beats: &[f64]) {
    let mut last = 0i64;
    for &b in beats {
        let t = (b * TICKS).round() as i64;
        let _ = write!(s, " {}", t - last);
        last = t;
    }
}

/// The beats in [`beats_text`]'s words.
pub(crate) fn parse_beats(words: &str) -> Option<Vec<f64>> {
    let mut t = 0i64;
    words
        .split_whitespace()
        .map(|d| {
            t += d.parse::<i64>().ok()?;
            Some(t as f64 / TICKS)
        })
        .collect()
}

/// An analysis as the cache file's text, under its `key`, in
/// [`cache::ANALYSIS`]'s version.
pub fn to_text(key: &str, a: &Analysis) -> String {
    cache::ANALYSIS.wrap(&analysis_text(key, a))
}

/// The cache file's text after its header.
fn analysis_text(key: &str, a: &Analysis) -> String {
    let mut s = format!("jodel-rhythm {VERSION}\nkey {}\n", key.replace('\n', " "));
    let _ = writeln!(s, "duration {:.4}", a.duration);
    let _ = writeln!(s, "confidence {:.4}", a.confidence);
    let _ = writeln!(s, "chosen {}", a.chosen);
    for g in &a.groupings {
        let _ = writeln!(
            s,
            "grouping {:.4} {} {}",
            g.start, g.grouping.pulses, g.grouping.phase
        );
    }
    let ids: Vec<&str> = a.methods.iter().map(|m| m.id()).collect();
    let _ = writeln!(s, "methods {}", ids.join(" "));
    let _ = writeln!(s, "stems {}", a.stems);
    match a.key {
        Some(k) if k.tagged => {
            let _ = writeln!(s, "tonality {} tag", k.class);
        }
        Some(k) => {
            let _ = writeln!(s, "tonality {} {:.4}", k.class, k.confidence);
        }
        None => {}
    }
    s.push_str("beats");
    beats_text(&mut s, &a.beats);
    s.push_str("\ndownbeats");
    for i in &a.downbeats {
        let _ = write!(s, " {i}");
    }
    s.push('\n');
    for r in &a.regions {
        let _ = writeln!(s, "region {:.4} {:.4} {:.4}", r.start, r.end, r.confidence);
    }
    for c in &a.candidates {
        let _ = writeln!(s, "candidate {:.3} {:.4} {}", c.bpm, c.agreement, c.name);
    }
    s
}

/// The analysis in a cache file's text, if it was stored under `key` in
/// [`cache::ANALYSIS`]'s version.
pub fn from_text(key: &str, text: &str) -> Option<Analysis> {
    let mut lines = cache::ANALYSIS.body(text)?.lines();
    if lines.next()? != format!("jodel-rhythm {VERSION}") {
        return None;
    }
    if lines.next()?.strip_prefix("key ")? != key.replace('\n', " ") {
        return None;
    }
    let mut a = Analysis::default();
    for line in lines {
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        match word {
            "duration" => a.duration = rest.parse().ok()?,
            "confidence" => a.confidence = rest.parse().ok()?,
            "chosen" => a.chosen = rest.to_owned(),
            "grouping" => {
                let mut parts = rest.split_whitespace();
                let start = parts.next()?.parse().ok()?;
                let pulses = parts.next()?.parse().ok()?;
                let phase = parts.next()?.parse().ok()?;
                a.groupings.push(Span {
                    start,
                    grouping: Grouping::new(pulses, phase)?,
                });
            }
            "stems" => a.stems = rest.parse().ok()?,
            "tonality" => {
                let (class, confidence) = rest.split_once(' ')?;
                let class: u8 = class.parse().ok()?;
                if class >= 24 {
                    return None;
                }
                let tagged = confidence == "tag";
                a.key = Some(jodel_rhythm::key::Key {
                    class,
                    confidence: if tagged {
                        1.0
                    } else {
                        confidence.parse().ok()?
                    },
                    tagged,
                });
            }
            "methods" => {
                for id in rest.split_whitespace() {
                    a.methods.push(Method::parse(id)?);
                }
            }
            "beats" => a.beats = parse_beats(rest)?,
            "downbeats" => {
                for i in rest.split_whitespace() {
                    a.downbeats.push(i.parse().ok()?);
                }
            }
            "region" => {
                let v: Vec<f64> = rest
                    .split_whitespace()
                    .map(|x| x.parse().ok())
                    .collect::<Option<_>>()?;
                let [start, end, confidence] = v[..] else {
                    return None;
                };
                a.regions.push(Region {
                    start,
                    end,
                    confidence,
                });
            }
            "candidate" => {
                let mut parts = rest.splitn(3, ' ');
                a.candidates.push(CandidateReport {
                    bpm: parts.next()?.parse().ok()?,
                    agreement: parts.next()?.parse().ok()?,
                    name: parts.next()?.to_owned(),
                });
            }
            _ => return None,
        }
    }
    a.derive();
    Some(a)
}

/// The analysis of `path` by `recipe` from the cache in `dir`, without its
/// waveform.
pub(super) fn load_analysis(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> Option<Analysis> {
    let key = cache_key(path, &recipe.methods, recipe.stems)?;
    let file = crate::mover::locate(&cache_file(dir, &key));
    from_text(&key, &fs::read_to_string(&file).ok()?)
}

/// The stored analysis of `path` by `recipe` and its waveform, if there
/// are both for the file as it is now.
pub fn load(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> Option<(Analysis, Waveform)> {
    let key = cache_key(path, &recipe.methods, recipe.stems)?;
    // Where it is now: while the data folder moves, not yet moved is where
    // it was.
    let file = crate::mover::locate(&cache_file(dir, &key));
    let analysis = from_text(&key, &fs::read_to_string(&file).ok()?)?;
    let waveform = Waveform::from_bytes(&fs::read(file.with_extension("wave")).ok()?)?;
    Some((analysis, waveform))
}

/// Stores the analysis of `path` and its waveform, under the methods it
/// used (the waveform first: an analysis without one is not loaded).
pub fn store(
    dir: &Path,
    path: &Path,
    analysis: &Analysis,
    waveform: &Waveform,
) -> std::io::Result<()> {
    let Some(key) = cache_key(path, &analysis.methods, analysis.stems) else {
        return Ok(());
    };
    // Where it goes now: while the data folder moves, into the new one.
    let dir = &crate::mover::current(dir);
    fs::create_dir_all(dir)?;
    let file = cache_file(dir, &key);
    durable::write_atomic(&file.with_extension("wave"), &waveform.to_bytes())?;
    durable::write_atomic(&file, to_text(&key, analysis).as_bytes())
}

/// Whether the analysis of `path` by `recipe` is in the cache in `dir`
/// (without reading it), and with stems, the stems themselves beside it
/// ([`crate::stems_cache::beside`]).
pub fn is_stored(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> bool {
    let analysed = cache_key(path, &recipe.methods, recipe.stems).is_some_and(|key| {
        let file = cache_file(dir, &key);
        file.is_file() && file.with_extension("wave").is_file()
    });
    analysed
        && (!recipe.stems || crate::stems_cache::is_kept(&crate::stems_cache::beside(dir), path))
}

#[cfg(test)]
mod tests {
    use jodel_rhythm::waveform::Lane;

    use super::*;

    #[test]
    fn an_analysis_round_trips_through_its_text_under_its_key_only() {
        let truth = jodel_rhythm::synth::beat_times(0.5, 19.5, |_| 120.0);
        let samples = jodel_rhythm::synth::band(20.0, &truth, 4, 0.5).samples;
        let mut a = jodel_rhythm::analyze_with(&samples, &[Method::Flux]);
        assert!(a.beats.len() > 30);
        a.key = Some(jodel_rhythm::key::Key {
            class: 7,
            confidence: 0.4375,
            tagged: false,
        });
        let text = to_text("song.flac 123 456 v1", &a);
        let back = from_text("song.flac 123 456 v1", &text).unwrap();
        assert_eq!(back.beats.len(), a.beats.len());
        assert!(back
            .beats
            .iter()
            .zip(&a.beats)
            .all(|(x, y)| (x - y).abs() < 1e-4));
        assert_eq!(back.downbeats, a.downbeats);
        assert_eq!(back.regions.len(), a.regions.len());
        assert_eq!(back.candidates.len(), a.candidates.len());
        assert_eq!(back.chosen, a.chosen);
        assert_eq!(back.groupings.len(), a.groupings.len());
        assert!(back
            .groupings
            .iter()
            .zip(&a.groupings)
            .all(|(x, y)| x.grouping == y.grouping && (x.start - y.start).abs() < 1e-4));
        assert_eq!(back.methods, vec![Method::Flux]);
        assert_eq!(back.key, a.key);
        a.key = jodel_rhythm::key::Key::from_tag("Ebm");
        let tagged = from_text("song.flac 123 456 v1", &to_text("song.flac 123 456 v1", &a));
        assert_eq!(tagged.unwrap().key, a.key);
        let (s, t) = (a.summary.unwrap(), back.summary.unwrap());
        assert!((s.bpm - t.bpm).abs() < 0.01);
        assert!(from_text("song.flac 123 999 v1", &text).is_none());
    }

    #[test]
    fn the_cache_follows_the_file() {
        let dir = std::env::temp_dir().join(format!("jodel-rhythm-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("track.wav");
        fs::write(&file, b"one").unwrap();
        let cache = dir.join("cache");
        let recipe = |methods: &[Method]| RhythmRecipe {
            methods: methods.to_vec(),
            stems: false,
        };
        let a = Analysis {
            beats: vec![0.5, 1.0, 1.5, 2.0, 2.5],
            confidence: 0.9,
            methods: vec![Method::Network],
            ..Analysis::default()
        };
        let wave = Waveform {
            mix: Lane {
                peak: vec![1, 2, 3],
                rms: vec![0, 1, 2],
            },
            stems: None,
            melody: None,
        };
        store(&cache, &file, &a, &wave).unwrap();
        let (back, back_wave) = load(&cache, &file, &recipe(&[Method::Network])).unwrap();
        assert_eq!(back.beats, a.beats);
        assert_eq!(back_wave, wave);
        // Other methods, or stems: analysed afresh.
        assert!(load(&cache, &file, &recipe(&Method::ALL)).is_none());
        let with_stems = RhythmRecipe {
            stems: true,
            ..recipe(&[Method::Network])
        };
        assert!(load(&cache, &file, &with_stems).is_none());
        // Another size: analysed afresh.
        fs::write(&file, b"longer").unwrap();
        assert!(load(&cache, &file, &recipe(&[Method::Network])).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// The names the caches keep a file's analysis, lyrics and stems by,
    /// byte for byte: the file's identity (`PATH SIZE NANOS`), the
    /// analysis's key after it, and the names hashed from them.
    #[test]
    fn the_cache_keys_are_pinned() {
        let dir = std::env::temp_dir().join(format!("jodel-keys-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("track.wav");
        fs::write(&file, b"one").unwrap();
        let modified = (fs::metadata(&file).unwrap().modified().unwrap())
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let identity = format!("{} 3 {modified}", file.display());
        assert_eq!(cache::identity(&file).as_deref(), Some(identity.as_str()));
        let key = cache_key(&file, &[Method::Flux, Method::Network], true).unwrap();
        assert_eq!(key, format!("{identity} v{VERSION} flux+network+stems"));
        assert_eq!(key_parts(&key), Some((identity.as_str(), VERSION)));
        let parts = cache::identity_parts(&identity);
        assert_eq!(parts, Some((file.as_path(), 3, modified)));
        let stems = crate::stems_cache::file_for(&dir, &file).unwrap();
        assert_eq!(stems, crate::stems_cache::file_of(&dir, &identity));
        let lyrics = crate::lyrics::store::cache_key(&file).unwrap();
        assert_eq!(lyrics, crate::lyrics::store::key_of(&identity));
        // The names, of a made-up file.
        let id = "song.flac 123 456";
        assert_eq!(crate::lyrics::store::key_of(id), "c28d16413b21d07b");
        let stems = crate::stems_cache::file_of(Path::new("s"), id);
        assert_eq!(stems, Path::new("s").join("d79ad5e0042c7176.stems"));
        let analysis = cache_file(Path::new("a"), "song.flac 123 456 v5 flux+network+stems");
        assert_eq!(analysis, Path::new("a").join("c587afc5db07616b.txt"));
        let _ = fs::remove_dir_all(&dir);
    }
}
