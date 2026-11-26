//! The queue changed while it plays ([`Command::Remove`], [`Command::Move`]):
//! playback goes on as it was, the tracks in their new order.

mod common;

use std::path::PathBuf;

use common::*;
use jodel_output::SampleFormat;
use jodel_playback::{Event, Player, PlayerOptions};
use jodel_testkit::TempDir;

const RATE: u32 = 44_100;

/// Three short tracks queued, the first playing (its first half second
/// written), and the player.
fn setup(
    name: &str,
) -> (
    TempDir,
    Vec<PathBuf>,
    Player,
    std::sync::mpsc::Receiver<Event>,
    std::sync::Arc<Gate>,
) {
    let dir = TempDir::new(name);
    let paths: Vec<PathBuf> = ["a", "b", "c"]
        .iter()
        .map(|n| {
            let path = dir.join(format!("{n}.wav"));
            ramp_wav(&path, RATE, RATE);
            path
        })
        .collect();
    let (target, _captures, gate) = GatedTarget::new(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let (player, events) = Player::new(Box::new(target), PlayerOptions::default());
    player.load(paths.clone()).unwrap();
    gate.allow(u64::from(RATE) / 2);
    player.play(0).unwrap();
    gate.wait_written(u64::from(RATE) / 2);
    (dir, paths, player, events, gate)
}

/// The last queue reported.
fn last_queue(events: &[Event]) -> (Vec<PathBuf>, Option<usize>) {
    events
        .iter()
        .rev()
        .find_map(|e| match e {
            Event::QueueChanged { queue, current } => Some((queue.clone(), *current)),
            _ => None,
        })
        .expect("a queue")
}

#[test]
fn a_track_moved_ahead_plays_next() {
    let (_dir, p, player, events, gate) = setup("queue-move");
    player.move_track(2, 1).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert_eq!(
        last_queue(&seen).0,
        [p[0].clone(), p[2].clone(), p[1].clone()]
    );
    assert_eq!(
        started_paths(&seen),
        [(0, p[0].clone()), (1, p[2].clone()), (2, p[1].clone())]
    );
}

#[test]
fn a_track_removed_is_not_played() {
    let (_dir, p, player, events, gate) = setup("queue-remove");
    player.remove(1).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert_eq!(
        last_queue(&seen),
        (vec![p[0].clone(), p[2].clone()], Some(0))
    );
    assert_eq!(started_paths(&seen), [(0, p[0].clone()), (1, p[2].clone())]);
}

#[test]
fn the_track_heard_removed_gives_way_to_the_next() {
    let (_dir, p, player, events, gate) = setup("queue-remove-heard");
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.remove(0).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 0,
            finished: false
        }
    )));
    assert_eq!(started_paths(&seen), [(0, p[1].clone()), (1, p[2].clone())]);
}

#[test]
fn the_track_heard_moved_plays_on_from_its_new_place() {
    let (_dir, p, player, events, gate) = setup("queue-move-heard");
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // The playing track to the end: it plays on, and the queue ends after
    // it (it is the last now).
    player.move_track(0, 2).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert_eq!(
        last_queue(&seen),
        (vec![p[1].clone(), p[2].clone(), p[0].clone()], Some(2))
    );
    assert!(
        started_paths(&seen).is_empty(),
        "{:?}",
        started_paths(&seen)
    );
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 2,
            finished: true
        }
    )));
    // Its positions carry its new index.
    assert!(seen
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) => Some(p.index),
            _ => None,
        })
        .next_back()
        .is_some_and(|i| i == 2));
}
