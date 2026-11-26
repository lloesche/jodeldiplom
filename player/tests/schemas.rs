//! The files kept on disk, pinned per schema version (see
//! `player/src/cache.rs`): what this player writes of each schema must be
//! exactly the sample of that version in `tests/schemas/`, and it must
//! read that sample back.
//!
//! A change to what a schema's files hold, or how, fails here. Then raise
//! the schema's version in `player/src/cache.rs` and run the tests with
//! `UPDATE_SCHEMAS=1`, which adds the new version's sample (it never
//! changes the sample of a version already there).

#![allow(
    clippy::disallowed_methods,
    reason = "tests: kept files as each schema version wrote them, and a new sample under tests/ when UPDATE_SCHEMAS asks"
)]

use std::path::{Path, PathBuf};

use jodel_lyrics::{Found, Lyrics, Source};
use jodel_rhythm::meter::{Grouping, Span};
use jodel_rhythm::waveform::{Lane, Waveform};
use jodel_rhythm::{Analysis, CandidateReport, Method, Region};
use jodeldiplom::cache::{self, Schema};
use jodeldiplom::core::grid::Grid;
use jodeldiplom::lyrics::store::{Cache, Kept};
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

/// The text of a file with the header taken off: as files were written
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
    assert_eq!(rhythm::from_text(key, &kept), Some(a));
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
    // As found: its words, and when the source synced its lines (no line
    // or word times of their own).
    let found = Lyrics::parse_text(
        "[00:01.00]One line\n[00:03.50]Two\n[00:05.00]\nThree\n",
        Source::Lrclib,
    );
    c.store_text("k", &found).unwrap();
    let kept = pinned(&cache::LYRICS_TEXT, "lrc", &read("k.lrc"));
    let lines = |l: &Lyrics| -> Vec<String> { l.lines.iter().map(|l| l.text()).collect() };
    std::fs::write(dir.join("k.lrc"), &kept).unwrap();
    let text = |c: &Cache| match c.text("k") {
        Kept::Current(l) => l,
        other => panic!("not read: {other:?}"),
    };
    let back = text(&c);
    assert_eq!(lines(&back), ["One line", "Two", "", "Three"]);
    assert_eq!(back.lines, found.lines);
    assert!(back.lines.iter().all(|l| l.time.is_none()));
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
    // When the source synced its lines is the text's, not kept with it.
    timed.lines[0].synced = found.lines[0].synced;
    timed.lines[0].words[0].letters = vec![0.0, 0.1, 0.3];
    timed.lines[0].words[1].letters = vec![0.0, 0.04, 0.2, 0.5];
    c.store_timed("t", &timed).unwrap();
    let kept = pinned(&cache::LYRICS_TIMED, "lrc", &read("t.timed.lrc"));
    std::fs::write(dir.join("t.timed.lrc"), &kept).unwrap();
    let back = c.timed("t").unwrap();
    timed.lines[0].synced = None;
    assert_eq!(back.to_lrc(), timed.to_lrc());
    assert_eq!(back.lines[0].words[1].letters.len(), 4);
    assert_eq!(back.lines[0].synced, None);
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
    use jodeldiplom::scores::{from_text, to_text, Best, Bests};
    let mut bests = Bests::new();
    bests.insert(
        "/music/a song.flac".into(),
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

#[test]
fn a_file_without_a_header_is_the_users_first_version_or_no_cache() {
    // The user's own work: version 1, as written before there were headers.
    let grid = Grid {
        beats: vec![0.25, 0.75, 1.25],
        downbeats: vec![0],
    };
    let kept = grids::to_text("k", &grid);
    assert_eq!(
        grids::from_text("k", &headerless(kept.as_bytes())),
        Some(grid)
    );
    for schema in [cache::SCORES, cache::DELAYS] {
        let first = std::fs::read(sample(&schema, 1, "txt")).unwrap();
        let text = headerless(&first);
        let read = match schema.name {
            "scores" => jodeldiplom::scores::from_text(&text).len(),
            _ => jodeldiplom::delays::from_text(&text).len(),
        };
        assert!(read > 0, "{}", schema.name);
    }
    // A cache's: as if there were none, made again (never an error).
    let key = "/music/a.flac 1 2 v1";
    let analysis = rhythm::to_text(key, &Analysis::default());
    assert!(rhythm::from_text(key, &analysis).is_some());
    assert_eq!(
        rhythm::from_text(key, &headerless(analysis.as_bytes())),
        None
    );
    let dir = temp_dir("headerless");
    let c = Cache::new(&dir);
    let found = Lyrics::parse_text("One line\n", Source::Lrclib);
    c.store_text("k", &found).unwrap();
    c.store_timed("k", &found).unwrap();
    for name in ["k.lrc", "k.timed.lrc"] {
        let file = dir.join(name);
        let text = headerless(&std::fs::read(&file).unwrap());
        durable::write_atomic(&file, text.as_bytes()).unwrap();
    }
    let text = c.text("k");
    assert!(matches!(text, Kept::Outdated), "{text:?}");
    assert_eq!(c.timed("k"), None);
    let _ = std::fs::remove_dir_all(&dir);
}

fn small_layers() -> jodeldiplom::cover::CoverLayers {
    use jodeldiplom::cover::{CoverLayers, NearMap, Swatch};
    let near = |v: u16| NearMap {
        width: 2,
        height: 2,
        near: vec![0, v, 30_000, 65_535],
    };
    CoverLayers {
        plate: image::RgbImage::from_fn(4, 4, |x, y| image::Rgb([x as u8 * 60, y as u8 * 60, 9])),
        subject: Some(image::RgbaImage::from_fn(4, 4, |x, y| {
            image::Rgba([200, 10, 20, (x * y * 28) as u8])
        })),
        depth: near(1_000),
        plate_depth: near(500),
        outline: image::GrayImage::from_fn(4, 4, |x, _| image::Luma([x as u8 * 80])),
        palette: vec![
            Swatch {
                rgb: [200, 10, 20],
                share: 0.6,
            },
            Swatch {
                rgb: [0, 60, 9],
                share: 0.4,
            },
        ],
    }
}

#[test]
fn cover_files_are_pinned() {
    use jodeldiplom::cover::{from_bytes, to_bytes, CoverStore, Perceptual};
    // The layout pinned, whatever made it: a fixed maker here.
    let layers = jodeldiplom::cover::Kept {
        made: "a-maker".into(),
        thumb: image::RgbImage::from_fn(2, 2, |x, y| image::Rgb([x as u8, y as u8, 7])),
        layers: small_layers(),
    };
    let kept = pinned(&cache::COVER_LAYERS, "bin", &to_bytes(&layers));
    assert_eq!(from_bytes(&kept), Some(layers));
    // A song's entry and its album's covers, as the store writes them
    // (fixed hashes: the files' layout is pinned, not the hashing).
    let dir = temp_dir("covers");
    let store = CoverStore::new(&dir);
    let seen = Perceptual {
        bits: 0x3626_c7de_a7b6_c100,
        mean: [115, 103, 158],
    };
    store
        .link("0123456789abcdef", Some("fc0d153cdfbefb992e61"))
        .unwrap();
    store
        .add_to_album("a0123456789abcdef", "fc0d153cdfbefb992e61", &seen)
        .unwrap();
    let read = |p: &str| std::fs::read(dir.join(p)).unwrap();
    let kept = pinned(
        &cache::COVER_SONG,
        "txt",
        &read("songs/0123456789abcdef.txt"),
    );
    std::fs::write(dir.join("songs/0123456789abcdef.txt"), &kept).unwrap();
    assert_eq!(
        store.linked("0123456789abcdef"),
        Some(Some("fc0d153cdfbefb992e61".to_owned()))
    );
    let kept = pinned(
        &cache::COVER_ALBUM,
        "txt",
        &read("albums/a0123456789abcdef.txt"),
    );
    std::fs::write(dir.join("albums/a0123456789abcdef.txt"), &kept).unwrap();
    assert_eq!(
        store.album("a0123456789abcdef"),
        vec![("fc0d153cdfbefb992e61".to_owned(), seen)]
    );
}

/// A cache file of the original singer's phrases: pinned, read back the
/// same; one measured another way (its measuring's version) not read.
#[test]
fn shine_files_are_pinned() {
    use jodeldiplom::core::shine::{Par, Phrase, Reference, Yardstick};
    use jodeldiplom::shine_cache;
    let reference = Reference {
        phrases: vec![
            None,
            Some(Phrase {
                yardstick: Yardstick {
                    word_hits: vec![12.0, 30.0, 2.0, 7.0],
                    voiced: 0.4125,
                    off: Some(21.5),
                    spread: None,
                    held: vec![0.8, 1.0],
                    range: Some(0.75),
                    mask: [vec![false; 12], vec![true; 30], vec![false; 5]].concat(),
                },
                par: Par {
                    shine: 0.97,
                    pitch: 0.915,
                },
            }),
        ],
    };
    // The measuring's own version is apart from the schema's.
    let measured = format!(
        "\nversion {}\n",
        jodeldiplom::core::shine::reference::VERSION
    );
    let text = shine_cache::to_text(&reference).replace(&measured, "\nversion x\n");
    let kept = pinned(&cache::SHINE, "txt", text.as_bytes());
    let kept = String::from_utf8(kept)
        .unwrap()
        .replace("\nversion x\n", &measured);
    assert_eq!(shine_cache::from_text(&kept), Some(reference));
    let other = kept.replace("\nversion ", "\nversion 9");
    assert_eq!(shine_cache::from_text(&other), None);
}

/// A song's loudness kept: pinned, read back the same.
#[test]
fn loudness_files_are_pinned() {
    use jodeldiplom::loudness::{self, Measured, Played};
    let measured = Measured {
        lufs: Some(-14.96),
        from: Played::Stems {
            voice_db: f64::NEG_INFINITY,
        },
    };
    let kept = pinned(
        &cache::LOUDNESS,
        "txt",
        loudness::to_text(&measured).as_bytes(),
    );
    let kept = String::from_utf8(kept).unwrap();
    assert_eq!(loudness::from_text(&kept), Some(measured));
}

/// A bank of the game's sounds kept: pinned, read back bit for bit; one
/// of other sounds (another key) or at another rate not read.
#[test]
fn sfx_bank_files_are_pinned() {
    use jodeldiplom::sfx::store;
    let sounds: Vec<(String, Vec<f32>)> = vec![
        ("pop-0".into(), vec![0.25, -0.5, 1.0, -1.0]),
        ("break".into(), Vec::new()),
        ("tier-diva".into(), vec![f32::MIN_POSITIVE, 0.125]),
    ];
    let refs: Vec<(String, &[f32])> = sounds.iter().map(|s| (s.0.clone(), &s.1[..])).collect();
    let key = 0x0123_4567_89ab_cdef;
    let kept = pinned(
        &cache::SFX_BANK,
        "bin",
        &store::to_bytes(48_000, &refs, key),
    );
    assert_eq!(store::from_bytes(&kept, key, 48_000), Some(sounds));
    assert_eq!(store::from_bytes(&kept, key + 1, 48_000), None);
    assert_eq!(store::from_bytes(&kept, key, 44_100), None);
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
