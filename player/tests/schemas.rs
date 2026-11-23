//! The files kept on disk, pinned per schema version (see
//! `player/src/cache.rs`): what this player writes of each schema must be
//! exactly the sample of that version in `tests/schemas/`, and it must
//! read that sample back.
//!
//! A change to what a schema's files hold, or how, fails here. Then raise
//! the schema's version in `player/src/cache.rs` and run the tests with
//! `UPDATE_SCHEMAS=1`, which adds the new version's sample (it never
//! changes the sample of a version already there).

use std::path::{Path, PathBuf};

use jodel_lyrics::{Found, Lyrics, Source};
use jodel_rhythm::meter::{Grouping, Span};
use jodel_rhythm::waveform::{Lane, Waveform};
use jodel_rhythm::{Analysis, CandidateReport, Method, Region};
use jodeldiplom::cache::{self, Schema};
use jodeldiplom::core::grid::Grid;
use jodeldiplom::lyrics::Cache;
use jodeldiplom::{grids, rhythm};

/// The sample of `schema`'s version `version`.
fn sample(schema: &Schema, version: u32, extension: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/schemas")
        .join(format!("{}-{version}.{extension}", schema.name))
}

/// The player that wrote a header, made the same for the samples.
fn any_player(bytes: &[u8]) -> Vec<u8> {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.replace(cache::PLAYER, "jodeldiplom x").into_bytes(),
        Err(_) => bytes.to_vec(),
    }
}

/// `written` (a file of `schema` as this player writes it) against the
/// sample of its version; returns the sample.
fn pinned(schema: &Schema, extension: &str, written: &[u8]) -> Vec<u8> {
    let file = sample(schema, schema.version, extension);
    let written = any_player(written);
    match std::fs::read(&file) {
        Ok(kept) => {
            assert!(
                kept == written,
                "The {} files this player writes changed ({} differs).\n\
                 Raise its version in player/src/cache.rs, then run the tests \
                 with UPDATE_SCHEMAS=1 to add the new version's sample.\n\
                 Written now:\n{}",
                schema.name,
                file.display(),
                String::from_utf8_lossy(&written)
            );
            kept
        }
        Err(_) if std::env::var_os("UPDATE_SCHEMAS").is_some() => {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, &written).unwrap();
            written
        }
        Err(_) => panic!(
            "No sample of {} version {} ({}): run the tests with UPDATE_SCHEMAS=1 to add it",
            schema.name,
            schema.version,
            file.display()
        ),
    }
}

/// The text of a sample with the header taken off: as files were written
/// before there were headers (version 1).
fn headerless(text: &[u8]) -> String {
    let text = String::from_utf8(text.to_vec()).unwrap();
    text.split_once('\n').unwrap().1.to_owned()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("jodel-schemas-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn analysis_files_are_pinned() {
    let mut a = Analysis {
        duration: 12.5,
        beats: vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        downbeats: vec![0, 4, 8],
        confidence: 0.875,
        chosen: "flux/plp".into(),
        groupings: vec![Span {
            start: 0.5,
            grouping: Grouping::new(2, 0).unwrap(),
        }],
        methods: vec![Method::ALL[0]],
        stems: true,
        regions: vec![Region {
            start: 0.5,
            end: 4.5,
            confidence: 0.75,
        }],
        candidates: vec![CandidateReport {
            name: "flux/plp".into(),
            bpm: 120.0,
            agreement: 0.5,
        }],
        key: Some(jodel_rhythm::key::Key {
            class: 19,
            confidence: 0.625,
            tagged: false,
        }),
        ..Analysis::default()
    };
    a.derive();
    let key = "/music/a.flac 1234 5678 v9 flux+stems";
    // The analysis's own version is apart from the schema's.
    let analysed = format!("\njodel-rhythm {}\n", jodel_rhythm::VERSION);
    let written = rhythm::to_text(key, &a).replace(&analysed, "\njodel-rhythm x\n");
    let kept = pinned(&cache::ANALYSIS, "txt", written.as_bytes());
    let kept = String::from_utf8(kept)
        .unwrap()
        .replace("\njodel-rhythm x\n", &analysed);
    assert_eq!(rhythm::from_text(key, &kept), Some(a.clone()));
    if cache::ANALYSIS.version == 1 {
        assert_eq!(
            rhythm::from_text(key, &headerless(kept.as_bytes())),
            Some(a)
        );
    }
}

#[test]
fn waveform_files_are_pinned() {
    let lane = |k: u8| Lane {
        peak: vec![k, k + 1, k + 2],
        rms: vec![k / 2, k / 2 + 1, k / 2 + 2],
    };
    let w = Waveform {
        mix: lane(200),
        stems: Some([lane(10), lane(20), lane(30), lane(40)]),
        melody: Some(jodel_rhythm::waveform::Melody {
            cents: vec![0, 6400, 6450],
            sure: vec![0, 180, 255],
        }),
    };
    let kept = pinned(&cache::WAVEFORM, "bin", &w.to_bytes());
    assert_eq!(Waveform::from_bytes(&kept), Some(w));
}

#[test]
fn lyrics_files_are_pinned() {
    let dir = temp_dir("lyrics");
    let c = Cache::new(&dir);
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    // As found: its words only (the player keeps no times that came with
    // a text).
    let found = Lyrics::parse_text("[00:01.00]One line\n[00:03.50]Two\n", Source::Lrclib);
    c.store_text("k", &found).unwrap();
    let kept = pinned(&cache::LYRICS_TEXT, "lrc", &read("k.lrc"));
    let lines = |l: &Lyrics| -> Vec<String> { l.lines.iter().map(|l| l.text()).collect() };
    std::fs::write(dir.join("k.lrc"), &kept).unwrap();
    assert_eq!(lines(&c.text("k").unwrap()), ["One line", "Two"]);
    // None found.
    c.store_none("n", &Found::Instrumental, 1_700_000_000)
        .unwrap();
    let kept = pinned(&cache::LYRICS_NONE, "none", &read("n.none"));
    std::fs::write(dir.join("n.none"), &kept).unwrap();
    assert!(matches!(
        c.none("n", 1_700_000_100),
        Some(Found::Instrumental)
    ));
    // Timed by us, with where the letters were heard.
    let mut timed = Lyrics::parse(
        "[00:01.00]<00:01.00>One <00:01.40>line<00:02.00>\n",
        Source::Lrclib,
    );
    timed.timing.by = Some(jodel_lyrics::Maker::new("ctc:test", 2));
    timed.lines[0].words[0].letters = vec![0.0, 0.1, 0.3];
    timed.lines[0].words[1].letters = vec![0.0, 0.04, 0.2, 0.5];
    c.store_timed("t", &timed).unwrap();
    let kept = pinned(&cache::LYRICS_TIMED, "lrc", &read("t.timed.lrc"));
    std::fs::write(dir.join("t.timed.lrc"), &kept).unwrap();
    let back = c.timed("t").unwrap();
    assert_eq!(back.to_lrc(), timed.to_lrc());
    assert_eq!(back.lines[0].words[1].letters.len(), 4);
    if cache::LYRICS_TEXT.version == 1 {
        std::fs::write(dir.join("k.lrc"), headerless(&read("k.lrc"))).unwrap();
        assert_eq!(lines(&c.text("k").unwrap()), ["One line", "Two"]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn grid_files_are_pinned_and_every_version_is_read() {
    let grid = Grid {
        beats: vec![0.25, 0.75, 1.25, 1.75, 2.25],
        downbeats: vec![1],
    };
    let kept = pinned(&cache::GRID, "txt", grids::to_text("k", &grid).as_bytes());
    let kept = String::from_utf8(kept).unwrap();
    assert_eq!(grids::from_text("k", &kept), Some(grid.clone()));
    // The user's work: every version's sample is still read, and so is a
    // file from before there were headers.
    for version in 1..=cache::GRID.version {
        let text = std::fs::read_to_string(sample(&cache::GRID, version, "txt")).unwrap();
        assert!(grids::from_text("k", &text).is_some(), "version {version}");
    }
    assert_eq!(
        grids::from_text("k", &headerless(kept.as_bytes())),
        Some(grid)
    );
}

#[test]
fn score_files_are_pinned_and_every_version_is_read() {
    use jodeldiplom::core::singing::Level;
    use jodeldiplom::scores::{from_text, to_text, Best, Bests};
    let mut bests = Bests::new();
    bests.insert(
        ("/music/a song.flac".into(), Level::Medium),
        Best {
            points: 2300,
            hit: 23,
            words: 31,
        },
    );
    let kept = pinned(&cache::SCORES, "txt", to_text(&bests).as_bytes());
    assert_eq!(from_text(&String::from_utf8(kept).unwrap()), bests);
    // The singer's record: every version's sample is still read.
    for version in 1..=cache::SCORES.version {
        let text = std::fs::read_to_string(sample(&cache::SCORES, version, "txt")).unwrap();
        assert!(!from_text(&text).is_empty(), "version {version}");
    }
}

#[test]
fn delay_files_are_pinned_and_every_version_is_read() {
    use jodeldiplom::core::lag::{Delay, Delays};
    use jodeldiplom::delays::{from_text, to_text};
    let mut kept = Delays::new();
    kept.insert(
        (
            "{0.0.0.00000000}.{out}".into(),
            "{0.0.1.00000000}.{mic}".into(),
        ),
        Delay {
            measured_ms: Some(152),
            nudge_ms: Some(-20),
        },
    );
    kept.insert(
        ("Speakers".into(), "Built-in mic".into()),
        Delay {
            measured_ms: None,
            nudge_ms: Some(30),
        },
    );
    kept.insert(
        ("Speakers".into(), "USB mic".into()),
        Delay {
            measured_ms: Some(140),
            nudge_ms: None,
        },
    );
    let kept_text = pinned(&cache::DELAYS, "txt", to_text(&kept).as_bytes());
    assert_eq!(from_text(&String::from_utf8(kept_text).unwrap()), kept);
    // The singer's record: every version's sample is still read.
    for version in 1..=cache::DELAYS.version {
        let text = std::fs::read_to_string(sample(&cache::DELAYS, version, "txt")).unwrap();
        assert!(!from_text(&text).is_empty(), "version {version}");
    }
}

/// Every version of every schema keeps its sample.
#[test]
fn every_version_keeps_its_sample() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/schemas");
    for schema in cache::ALL {
        for version in 1..=schema.version {
            let found = std::fs::read_dir(&dir).unwrap().flatten().any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&format!("{}-{version}.", schema.name))
            });
            assert!(found, "no sample of {} version {version}", schema.name);
        }
    }
}
