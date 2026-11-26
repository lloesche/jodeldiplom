//! The file browser on a real folder tree: listing, lazy details for the
//! rows on screen, the filter, the recursive search and queueing a folder.
//! The app plays into a null output; no device is opened.

#![allow(
    clippy::disallowed_methods,
    reason = "tests: a fixture track copied into a temporary folder"
)]

mod common;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::browser::{self, EntryKind};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{browser_rows, row_entry, Action, Side};

const WAIT: Duration = Duration::from_secs(30);

fn start(dir: PathBuf, (cols, rows): (u16, u16)) -> App {
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = std::sync::Arc::new(|| Ok(Vec::new()));
    common::fitted_app(options, cols, rows)
}

fn names(app: &App) -> Vec<String> {
    browser_rows(&app.state)
        .into_iter()
        .filter_map(|r| row_entry(&app.state, r).map(|e| e.label().to_owned()))
        .collect()
}

#[test]
fn listing_puts_folders_first_and_hides_the_rest() {
    let tree = common::music_tree("list");
    let entries = browser::list_dir(&tree.root).unwrap();
    let names: Vec<(&str, EntryKind)> = entries.iter().map(|e| (e.name.as_str(), e.kind)).collect();
    assert_eq!(
        names,
        vec![
            ("..", EntryKind::Parent),
            ("Archive", EntryKind::Dir),
            ("Empty", EntryKind::Dir),
            ("Synthetic Ramps", EntryKind::Dir),
        ]
    );
    let album: Vec<String> = browser::list_dir(&tree.album)
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(
        album,
        vec![
            "..",
            "01 Ramp 44.1.flac",
            "02 Ramp 48.flac",
            "03 Chord 96k.wav",
            "10 Ramp again.flac"
        ]
    );
    let walked = browser::walk_playable(&tree.root);
    let walked: Vec<String> = walked
        .iter()
        .map(|p| {
            p.strip_prefix(&tree.root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert_eq!(
        walked,
        vec![
            "Archive/Deep/Hidden Track.flac",
            "Synthetic Ramps/01 Ramp 44.1.flac",
            "Synthetic Ramps/02 Ramp 48.flac",
            "Synthetic Ramps/03 Chord 96k.wav",
            "Synthetic Ramps/10 Ramp again.flac",
        ]
    );
    let mut found = Vec::new();
    browser::search(&tree.root, "RAMP", &AtomicBool::new(false), |hits| {
        found.extend(hits.into_iter().map(|e| e.relative.unwrap()))
    });
    assert_eq!(found.len(), 4, "{found:?}");
    assert!(found.contains(&"Synthetic Ramps".to_owned()));
    // A cancelled search stops.
    let mut none = Vec::new();
    browser::search(&tree.root, "a", &AtomicBool::new(true), |hits| {
        none.extend(hits)
    });
    assert!(none.is_empty());
    std::fs::remove_dir_all(&tree.root).unwrap();
}

#[test]
fn details_are_read_for_the_rows_on_screen() {
    let _guard = common::real_time();
    let tree = common::music_tree("details");
    let mut app = start(tree.album.clone(), (100, 30));
    assert!(common::wait_for(&mut app, WAIT, |a| {
        !a.state.browser.loading && a.state.details.len() == 4
    }));
    let chip = |name: &str, app: &App| app.state.details[&tree.album.join(name)].clone();
    assert_eq!(chip("01 Ramp 44.1.flac", &app).chip, "FLAC 44.1");
    assert_eq!(chip("02 Ramp 48.flac", &app).chip, "FLAC 48");
    assert_eq!(chip("03 Chord 96k.wav", &app).chip, "WAV 96");
    assert_eq!(chip("03 Chord 96k.wav", &app).seconds, Some(3.0));
    app.shutdown();

    // A folder longer than the screen: only what is shown is read.
    let big = common::temp_dir("big");
    for i in 0..40 {
        std::fs::copy(
            common::fixture(common::F44),
            big.join(format!("{i:02}.flac")),
        )
        .unwrap();
    }
    let mut app = start(big.clone(), (100, 16));
    // The rows the list shows in that window.
    let height = usize::from(app.state.fit.list_rows);
    assert!(common::wait_for(&mut app, WAIT, |a| {
        !a.state.browser.loading && a.state.pending.is_empty() && !a.state.details.is_empty()
    }));
    // The parent row takes one line.
    assert_eq!(app.state.details.len(), height - 1);
    assert!(app.state.details.contains_key(&big.join("00.flac")));
    assert!(!app.state.details.contains_key(&big.join("39.flac")));
    app.dispatch(Action::Op(Op::Bottom));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .details
        .contains_key(&big.join("39.flac"))));
    assert!(app.state.details.len() < 40);
    app.shutdown();
    std::fs::remove_dir_all(&tree.root).unwrap();
    std::fs::remove_dir_all(&big).unwrap();
}

#[test]
fn filter_then_search_below_then_queue_a_folder() {
    let _guard = common::real_time();
    let tree = common::music_tree("search");
    let mut app = start(tree.root.clone(), (100, 30));
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    app.dispatch(Action::Op(Op::Filter));
    for c in "hid".chars() {
        app.dispatch(Action::Type(c));
    }
    // Nothing in this folder is named like that ("hidden" folders are
    // not listed), so only the search row remains.
    assert!(names(&app).is_empty());
    app.dispatch(Action::Op(Op::SearchBelow));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .browser
        .search
        .as_ref()
        .is_some_and(|s| s.done)));
    assert_eq!(names(&app), vec!["Archive/Deep/Hidden Track.flac"]);
    // Back to the folder; queue the album folder with everything in it.
    app.dispatch(Action::Op(Op::Back));
    app.dispatch(Action::Op(Op::Back));
    let row = names(&app)
        .iter()
        .position(|n| n == "Synthetic Ramps")
        .unwrap();
    app.dispatch(Action::SelectRow(row + 1));
    app.dispatch(Action::Op(Op::Enqueue));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .queue
        .paths
        .len()
        == 4));
    let queued: Vec<String> = app
        .state
        .queue
        .paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        queued,
        vec![
            "01 Ramp 44.1.flac",
            "02 Ramp 48.flac",
            "03 Chord 96k.wav",
            "10 Ramp again.flac"
        ]
    );
    // The queue side reads its rows' details too.
    app.dispatch(Action::ShowSide(Side::Queue));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .queue
        .paths
        .iter()
        .all(|p| a.state.details.contains_key(p))));
    app.shutdown();
    std::fs::remove_dir_all(&tree.root).unwrap();
}

fn format_fixture(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/formats")
        .join(name)
}

/// The other formats: chips, tags and lengths from the headers alone, and
/// the same chips as a full read.
#[test]
fn rows_of_the_other_formats_come_from_their_headers() {
    for (name, chip, title) in [
        ("alac.m4a", "ALAC 44.1", Some("Format Test")),
        ("alac.caf", "ALAC 44.1", None),
        ("lossy.mp3", "MP3 96k", Some("Format Test")),
        ("lossy.ogg", "VORBIS 64k", Some("Format Test")),
        ("lossy.mp2", "MP2 64k", None),
        ("aac.m4a", "AAC 97k", Some("Format Test")),
    ] {
        let path = format_fixture(name);
        assert!(browser::is_playable(&path), "{name}");
        let quick = browser::quick_details(&path);
        assert_eq!(quick.chip, chip, "{name}");
        assert_eq!(quick.title.as_deref(), title, "{name}");
        let full = browser::read_details(&path);
        assert_eq!(full.chip, chip, "{name}");
        assert_eq!(quick.seconds, full.seconds, "{name}");
    }
    let quick = browser::quick_details(&format_fixture("lossy.mp3"));
    assert!((quick.seconds.unwrap() - 20_000.0 / 44_100.0).abs() < 1e-6);
    assert_eq!(quick.artist.as_deref(), Some("Fixture Band"));
    let quick = browser::quick_details(&format_fixture("aac.m4a"));
    assert!((quick.seconds.unwrap() - 20_000.0 / 44_100.0).abs() < 1e-6);
    // HE-AAC in an M4A file is listed, and its row says why it does not
    // play.
    let he = browser::quick_details(&format_fixture("he-aac.m4a"));
    assert_eq!(he.chip, "?");
    assert!(he
        .error
        .unwrap()
        .contains("HE-AAC (AAC with SBR) is not supported"));
}

/// The chips: lossy codecs with their bit rate, lossless ones with the
/// rate, PCM by its container.
#[test]
fn format_chips() {
    use jodel_playback::{Codec, Container, TrackInfo};
    let info = |container, codec, rate, bitrate| TrackInfo {
        path: PathBuf::from("/m/x"),
        container,
        codec,
        rate,
        channels: 2,
        bits: 24,
        float: false,
        bitrate,
        frames: None,
        tags: Default::default(),
        cover: None,
    };
    let chip = |i: &TrackInfo| browser::chip(i);
    assert_eq!(
        chip(&info(Container::Flac, Codec::Flac, 44_100, None)),
        "FLAC 44.1"
    );
    assert_eq!(
        chip(&info(Container::Wav, Codec::Pcm, 96_000, None)),
        "WAV 96"
    );
    assert_eq!(
        chip(&info(Container::Aiff, Codec::Pcm, 96_000, None)),
        "AIFF 96"
    );
    assert_eq!(
        chip(&info(Container::Caf, Codec::Pcm, 192_000, None)),
        "CAF 192"
    );
    assert_eq!(
        chip(&info(Container::Mp4, Codec::Alac, 44_100, None)),
        "ALAC 44.1"
    );
    assert_eq!(
        chip(&info(Container::Caf, Codec::Alac, 88_200, None)),
        "ALAC 88.2"
    );
    assert_eq!(
        chip(&info(Container::Mpeg, Codec::Mp3, 44_100, Some(320_000))),
        "MP3 320k"
    );
    assert_eq!(
        chip(&info(Container::Mpeg, Codec::Mp3, 44_100, Some(190_400))),
        "MP3 190k"
    );
    assert_eq!(
        chip(&info(Container::Mpeg, Codec::Mp3, 48_000, None)),
        "MP3 48"
    );
    assert_eq!(
        chip(&info(Container::Ogg, Codec::Vorbis, 44_100, Some(160_000))),
        "VORBIS 160k"
    );
    assert_eq!(
        chip(&info(Container::Mp4, Codec::Aac, 44_100, Some(256_000))),
        "AAC 256k"
    );
}
