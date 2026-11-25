//! What an analysis tells of itself: how long its parts took and what its
//! GPU work cost ([`Timings`]), and the last few analyses kept as text for
//! `--diag` ([`describe`], [`remember`], [`recent`]).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Duration;

use jodel_rhythm::Analysis;

use super::Listened;
use crate::core::RhythmRecipe;

/// How long the parts of hearing a file took.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    /// Decoding the file and feeding the beat methods (and the stems).
    pub decode: Duration,
    /// What separated the stems (the GPU's name), when they were asked for
    /// and there was one.
    pub separator: Option<String>,
    /// Separating the stems, start to finish (alongside the decoding).
    pub stems: Option<Duration>,
    /// Waiting for the stems once the file was decoded.
    pub stems_waited: Option<Duration>,
    /// What the separator's GPU work cost meanwhile (other work on it at
    /// the same time included), when the stems were separated.
    pub stems_gpu: Option<jodel_gpu::pace::Tally>,
    /// From the end of decoding until the evidence was in: the beat
    /// network and the cues, alongside the rest of the stems.
    pub listen: Duration,
    /// What the beat network's GPU work cost, where it ran on the GPU.
    pub network_gpu: Option<jodel_gpu::pace::Tally>,
}

/// The file in the cache folder holding the last analyses
/// ([`remember`]).
const RECENT: &str = "recent-analyses.txt";
/// How many analyses it keeps.
pub const RECENT_KEPT: usize = 3;
/// Between two analyses in the file.
const RECENT_BREAK: &str = "\n---\n";

/// The last analyses the player made (newest first), as kept in `dir`
/// for `--diag`: each a few lines of text.
pub fn recent(dir: &Path) -> Vec<String> {
    fs::read_to_string(crate::mover::locate(&dir.join(RECENT)))
        .map(|text| {
            text.split(RECENT_BREAK)
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Keeps `entry` as the newest of the last [`RECENT_KEPT`] analyses in
/// `dir`.
pub fn remember(dir: &Path, entry: &str) -> std::io::Result<()> {
    let mut entries = recent(dir);
    entries.insert(0, entry.trim().to_owned());
    entries.truncate(RECENT_KEPT);
    let dir = &crate::mover::current(dir);
    fs::create_dir_all(dir)?;
    durable::write_atomic(&dir.join(RECENT), entries.join(RECENT_BREAK).as_bytes())
}

/// Seconds as `1.23 s`.
fn seconds(d: Duration) -> String {
    format!("{:.2} s", d.as_secs_f64())
}

/// An analysis as [`remember`] keeps it: when (seconds since 1970), the
/// file's name (not its folders), how it was heard, what was found, how
/// long each part took and what the GPU work cost (the stems', the beat
/// network's).
pub fn describe(
    at: u64,
    path: &Path,
    recipe: &RhythmRecipe,
    listened: &Listened,
    analysis: &Analysis,
    decide: Duration,
    total: Duration,
) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let minutes = analysis.duration / 60.0;
    let mut s = format!(
        "at {at}\ntrack: {name} ({}:{:02})\n",
        minutes.floor(),
        (analysis.duration % 60.0).floor()
    );
    let ids: Vec<&str> = recipe.methods.iter().map(|m| m.id()).collect();
    let heard = match (
        &listened.timings.separator,
        &listened.stems_failed,
        analysis.stems,
    ) {
        (_, Some(e), _) => format!("the mix (the stems failed: {e})"),
        (Some(gpu), None, true) => format!("stems separated on {gpu}"),
        (None, None, _) if recipe.stems => "the mix (no GPU separates stems)".into(),
        _ => "the mix (stems off)".into(),
    };
    let _ = writeln!(s, "methods: {}; the cues heard {heard}", ids.join(", "));
    let tempo = match analysis.summary {
        Some(t) if t.steady => format!("{:.2} BPM steady", t.bpm),
        Some(t) => format!("{:.1} BPM ({:.1}–{:.1})", t.bpm, t.low, t.high),
        None => "no beat".into(),
    };
    let key = analysis.key.map_or("no key".to_owned(), |k| {
        if k.tagged {
            format!("{} ({}, from its tags)", k.name(), k.camelot())
        } else {
            format!(
                "{} ({}, {:.0} %)",
                k.name(),
                k.camelot(),
                100.0 * k.confidence
            )
        }
    });
    let _ = writeln!(
        s,
        "found: {tempo}, confidence {:.2}, chosen {}, {} beats, {key}",
        analysis.confidence,
        analysis.chosen,
        analysis.beats.len()
    );
    let t = &listened.timings;
    let stems = match (t.stems, t.stems_waited) {
        (Some(took), Some(waited)) => {
            format!(
                ", stems {} (waited {} after decoding)",
                seconds(took),
                seconds(waited)
            )
        }
        _ => String::new(),
    };
    let _ = write!(
        s,
        "took: decoding {}{stems}, listening {}, deciding {}, all {}",
        seconds(t.decode),
        seconds(t.listen),
        seconds(decide),
        seconds(total)
    );
    let gpu: Vec<String> = [
        ("the stems", t.stems_gpu),
        ("the beat network", t.network_gpu),
    ]
    .into_iter()
    .filter_map(|(what, tally)| Some(format!("{what} {}", tally?)))
    .collect();
    if !gpu.is_empty() {
        let _ = write!(s, "\nGPU: {}", gpu.join("; "));
    }
    s
}
